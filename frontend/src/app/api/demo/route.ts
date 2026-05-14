import { NextResponse } from "next/server";

import { gateSecret } from "@/lib/auth/demo-gate";
import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoModelChoice,
  DemoProvider,
  LiveDemoRunState,
  DemoRunMode,
  DemoSandboxMode
} from "@/lib/demo/demo-run-types";
import { backendBaseUrl, BACKEND_GET_TIMEOUT_MS, enrichOrTimeout } from "@/lib/demo/server-run";

export const runtime = "nodejs";

// Hard cap on an uploaded PDF — mirrors the "max 50 MB" the lab UI
// advertises. Checked from the Content-Length header so an oversized
// upload is rejected before any body is streamed.
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

function demoSecretHeaders(): Record<string, string> {
  const secret = gateSecret();
  return secret ? { "x-demo-secret": secret } : {};
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

function toModelChoice(request: Request): DemoModelChoice | undefined {
  const value = search(request).get("model");
  return value === "sonnet" || value === "opus" ? value : undefined;
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

export async function GET(request: Request) {
  const projectId = toProjectId(request);
  const params = backendQuery(request);
  const endpoint = projectId
    ? `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}`
    : `${backendBaseUrl()}/runs/latest${params.size ? `?${params}` : ""}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), BACKEND_GET_TIMEOUT_MS);
  try {
    const response = await fetch(endpoint, { cache: "no-store", signal: controller.signal });
    if (!response.ok) {
      return jsonFromBackend(response);
    }
    const state = (await response.json()) as LiveDemoRunState | null;
    if (!state || !state.projectId) {
      return NextResponse.json(state, { status: response.status });
    }
    const enriched = await enrichOrTimeout(state);
    return NextResponse.json(enriched, { status: response.status });
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
      // Stream the multipart body straight through to the backend. We do
      // NOT call request.formData() here: undici's multipart parser
      // (Node 21) throws "Failed to parse body as FormData" on real-size
      // browser uploads, and it throws *fast* — responding before the
      // browser has finished sending the body, which leaves the HTTP
      // connection half-written and poisons it (Chrome then reports
      // ERR_ALPN_NEGOTIATION_FAILED on reuse). A proxy has no business
      // parsing a body it only forwards: stream request.body through, so
      // the body is fully drained before we respond. The backend's
      // /runs/upload validates the PDF (missing / non-.pdf / empty ->
      // 400) and that response passes straight back via jsonFromBackend.
      const declaredBytes = Number(request.headers.get("content-length") ?? 0);
      if (declaredBytes > MAX_UPLOAD_BYTES) {
        return NextResponse.json(
          { error: "PDF is too large — the lab accepts uploads up to 50 MB." },
          { status: 413 }
        );
      }
      return jsonFromBackend(
        await fetch(`${backendBaseUrl()}/runs/upload`, {
          method: "POST",
          headers: { "content-type": contentType, ...demoSecretHeaders() },
          body: request.body,
          duplex: "half"
        } as RequestInit & { duplex: "half" })
      );
    }

    return jsonFromBackend(
      await fetch(`${backendBaseUrl()}/runs`, {
        method: "POST",
        headers: { "content-type": "application/json", ...demoSecretHeaders() },
        body: JSON.stringify({
          mode: toRunMode(request) ?? "offline",
          provider: toProvider(request) ?? "anthropic",
          verificationProvider: toVerificationProvider(request),
          executionMode: toExecutionMode(request) ?? "efficient",
          sandbox: toSandboxMode(request) ?? "runpod",
          gpuMode: toGpuMode(request) ?? "auto",
          model: toModelChoice(request) ?? "sonnet"
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
