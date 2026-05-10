import { NextResponse } from "next/server";

import type {
  DemoExecutionMode,
  DemoGpuMode,
  DemoProvider,
  DemoSandboxMode
} from "@/lib/demo/demo-run-types";
import { loadDemoRun, startDemoRun, stopDemoRun } from "@/lib/demo/node-runner";

export const runtime = "nodejs";

function toRunMode(request: Request): "offline" | "sdk" | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("mode");
  if (value === "offline" || value === "sdk") {
    return value;
  }
  return undefined;
}

function toProjectId(request: Request): string | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("projectId");
  return value || undefined;
}

function toProvider(request: Request): DemoProvider | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("provider");
  if (value === "anthropic" || value === "openai") {
    return value;
  }
  return undefined;
}

function toVerificationProvider(request: Request): DemoProvider | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("verificationProvider");
  if (value === "anthropic" || value === "openai") {
    return value;
  }
  return undefined;
}

function toExecutionMode(request: Request): DemoExecutionMode | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("executionMode");
  if (value === "efficient" || value === "max") {
    return value;
  }
  return undefined;
}

function toSandboxMode(request: Request): DemoSandboxMode | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("sandbox");
  if (value === "auto" || value === "docker" || value === "local" || value === "runpod") {
    return value;
  }
  return undefined;
}

function toGpuMode(request: Request): DemoGpuMode | undefined {
  const url = new URL(request.url);
  const value = url.searchParams.get("gpuMode");
  if (value === "off" || value === "auto" || value === "prefer" || value === "max") {
    return value;
  }
  return undefined;
}

export async function GET(request: Request) {
  const latest = await loadDemoRun(
    toProjectId(request),
    toRunMode(request),
    toProvider(request),
    toExecutionMode(request),
    toSandboxMode(request),
    toVerificationProvider(request),
    toGpuMode(request)
  );
  return NextResponse.json(latest);
}

export async function POST(request: Request) {
  try {
    const contentType = request.headers.get("content-type") ?? "";
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const paper = formData.get("paper");
      const mode = formData.get("mode");
      const provider = formData.get("provider");
      const verificationProvider = formData.get("verificationProvider");
      const executionMode = formData.get("executionMode");
      const sandboxMode = formData.get("sandbox");
      const gpuMode = formData.get("gpuMode");
      const runMode = mode === "sdk" || mode === "offline" ? mode : "offline";
      const llmProvider =
        provider === "anthropic" || provider === "openai"
          ? provider
          : toProvider(request) ?? "anthropic";
      const runVerificationProvider =
        verificationProvider === "anthropic" || verificationProvider === "openai"
          ? verificationProvider
          : toVerificationProvider(request);
      const runExecutionMode =
        executionMode === "efficient" || executionMode === "max"
          ? executionMode
          : toExecutionMode(request) ?? "efficient";
      const runSandboxMode =
        sandboxMode === "auto" ||
        sandboxMode === "docker" ||
        sandboxMode === "local" ||
        sandboxMode === "runpod"
          ? sandboxMode
          : toSandboxMode(request) ?? "auto";
      const runGpuMode =
        gpuMode === "off" || gpuMode === "auto" || gpuMode === "prefer" || gpuMode === "max"
          ? gpuMode
          : toGpuMode(request) ?? "auto";

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

      const bytes = new Uint8Array(await paper.arrayBuffer());
      const run = await startDemoRun(
        runMode,
        llmProvider,
        runExecutionMode,
        runSandboxMode,
        {
          uploadedPaper: {
            fileName: paper.name,
            bytes
          },
          verificationProvider: runVerificationProvider,
          gpuMode: runGpuMode
        }
      );
      return NextResponse.json(run, { status: 202 });
    }

    const run = await startDemoRun(
      toRunMode(request) ?? "offline",
      toProvider(request) ?? "anthropic",
      toExecutionMode(request) ?? "efficient",
      toSandboxMode(request) ?? "auto",
      {
        verificationProvider: toVerificationProvider(request),
        gpuMode: toGpuMode(request) ?? "auto"
      }
    );
    return NextResponse.json(run, { status: 202 });
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
    const run = await stopDemoRun(
      toRunMode(request) ?? "offline",
      toProjectId(request),
      toProvider(request),
      toExecutionMode(request),
      toSandboxMode(request),
      toVerificationProvider(request),
      toGpuMode(request)
    );
    return NextResponse.json(run);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to stop demo pipeline";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
