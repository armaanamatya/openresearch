import { NextResponse } from "next/server";

import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoRunMode,
  DemoSandboxMode
} from "@/lib/demo/demo-run-types";

export const runtime = "nodejs";

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}
function search(request: Request): URLSearchParams {
  return new URL(request.url).searchParams;
}

function toRunMode(request: Request): DemoRunMode | undefined {
  const value = search(request).get("mode");
  return value === "offline" || value === "sdk" ? value : undefined;
}

function toProjectId(request: Request): string | undefined {
  return search(request).get("projectId") || undefined;
}

function toProvider(request: Request): DemoProvider | undefined {
  const value = search(request).get("provider");
  return value === "anthropic" || value === "openai" ? value : undefined;
}

function toVerificationProvider(request: Request): DemoProvider | undefined {
  const value = search(request).get("verificationProvider");
  return value === "anthropic" || value === "openai" ? value : undefined;
}

function toExecutionMode(request: Request): DemoExecutionMode | undefined {
  const value = search(request).get("executionMode");
  return value === "efficient" || value === "max" ? value : undefined;
}

function toSandboxMode(request: Request): DemoSandboxMode | undefined {
  const value = search(request).get("sandbox");
  return value === "auto" || value === "docker" || value === "local" || value === "runpod"
    ? value
    : undefined;
}

function toGpuMode(request: Request): DemoGpuMode | undefined {
  const value = search(request).get("gpuMode");
  return value === "off" || value === "auto" || value === "prefer" || value === "max"
    ? value
    : undefined;
}

function backendQuery(request: Request): URLSearchParams {
  const params = new URLSearchParams();
  const mode = toRunMode(request);
  const provider = toProvider(request);
  const executionMode = toExecutionMode(request);
  const sandbox = toSandboxMode(request);
  const verificationProvider = toVerificationProvider(request);
  const gpuMode = toGpuMode(request);
  if (mode) params.set("mode", mode);
  if (provider) params.set("provider", provider);
  if (executionMode) params.set("executionMode", executionMode);
  if (sandbox) params.set("sandbox", sandbox);
  if (verificationProvider) params.set("verificationProvider", verificationProvider);
  if (gpuMode) params.set("gpuMode", gpuMode);
  return params;
}

async function jsonFromBackend(response: Response): Promise<NextResponse> {
  const text = await response.text();
  return new NextResponse(text || "null", {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/json"
    }
  });
}

// Backend GET timeout for the lab page's status polling. Single-worker
// uvicorn (--reload) blocks on long SSE streams, so we cap how long the
// browser waits and let the client retry with backoff.
const BACKEND_GET_TIMEOUT_MS = 4000;

export async function GET(request: Request) {
  const projectId = toProjectId(request);
  const params = backendQuery(request);
  const endpoint = projectId
    ? `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}`
    : `${backendBaseUrl()}/runs/latest${params.size ? `?${params}` : ""}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), BACKEND_GET_TIMEOUT_MS);
  try {
    return jsonFromBackend(
      await fetch(endpoint, { cache: "no-store", signal: controller.signal })
    );
  } catch (error) {
    const aborted =
      error instanceof Error && (error.name === "AbortError" || error.name === "TimeoutError");
    return NextResponse.json(
      {
        error: aborted ? "Backend timed out" : "Backend unreachable",
        code: aborted ? "backend_timeout" : "backend_unreachable"
      },
      { status: 504 }
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get("content-type") ?? "";
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const paper = formData.get("paper");
      if (!(paper instanceof File) || paper.size === 0) {
        return NextResponse.json(
          { error: "Upload a PDF before starting a lab run." },
          { status: 400 }
        );
      }
      const looksLikePdf =
        paper.type === "application/pdf" || paper.name.toLowerCase().endsWith(".pdf");
      if (!looksLikePdf) {
        return NextResponse.json(
          { error: "Only PDF uploads are supported in the lab right now." },
          { status: 400 }
        );
      }
      return jsonFromBackend(
        await fetch(`${backendBaseUrl()}/runs/upload`, {
          method: "POST",
          body: formData
        })
      );
    }

    return jsonFromBackend(
      await fetch(`${backendBaseUrl()}/runs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          mode: toRunMode(request) ?? "offline",
          provider: toProvider(request) ?? "anthropic",
          verificationProvider: toVerificationProvider(request),
          executionMode: toExecutionMode(request) ?? "efficient",
          sandbox: toSandboxMode(request) ?? "runpod",
          gpuMode: toGpuMode(request) ?? "auto"
        })
      })
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Demo pipeline failed";
    const statusCode =
      error &&
      typeof error === "object" &&
      "statusCode" in error &&
      typeof error.statusCode === "number"
        ? error.statusCode
        : 500;
    const code =
      error && typeof error === "object" && "code" in error
        ? String(error.code)
        : "demo_pipeline_failed";
    return NextResponse.json({ error: message, code }, { status: statusCode });
  }
}

export async function DELETE(request: Request) {
  try {
    let projectId = toProjectId(request);
    if (!projectId) {
      const params = backendQuery(request);
      const latest = await fetch(
        `${backendBaseUrl()}/runs/latest${params.size ? `?${params}` : ""}`,
        { cache: "no-store" }
      );
      if (!latest.ok) {
        return jsonFromBackend(latest);
      }
      const body = (await latest.json()) as { projectId?: string };
      projectId = body.projectId;
    }
    if (!projectId) {
      return NextResponse.json({ error: "Run not found" }, { status: 404 });
    }
    return jsonFromBackend(
      await fetch(`${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}`, {
        method: "DELETE"
      })
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to stop demo pipeline";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
