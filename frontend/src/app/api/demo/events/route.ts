import { createHash } from "crypto";
import { NextResponse } from "next/server";

import type { LiveDemoRunState } from "@/lib/demo/demo-run-types";
import { enrichRunStateWithPayload } from "@/lib/demo/server-payload";

export const runtime = "nodejs";

const ENRICH_TIMEOUT_MS = 250;
const SYNTHETIC_IDLE_MS = 3000;
const SYNTHETIC_INTERVAL_MS = 1000;

interface ParsedSseFrame {
  id: string | null;
  event: string;
  data: string;
}

class EnrichmentTimeoutError extends Error {
  constructor() {
    super("Timed out while enriching run state");
    this.name = "EnrichmentTimeoutError";
  }
}

function backendBaseUrl(): string {
  return (process.env.REPROLAB_BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

function hashJson(json: string): string {
  return createHash("sha1").update(json).digest("hex");
}

// Hash a stable subset of an enriched LiveDemoRunState. `generatedAt` and any
// future timestamp fields would otherwise force a synthetic emit on every tick
// even when the underlying state is identical.
const VOLATILE_HASH_KEYS = new Set(["generatedAt", "lastUpdated", "timestamp"]);
function stableEnrichedHash(state: LiveDemoRunState): string {
  const json = JSON.stringify(state, (key, value) =>
    VOLATILE_HASH_KEYS.has(key) ? undefined : value
  );
  return hashJson(json);
}

function parseSseFrame(frame: string): ParsedSseFrame {
  const lines = frame.split(/\r?\n/);
  const dataLines: string[] = [];
  let id: string | null = null;
  let event = "message";

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    const separatorIndex = line.indexOf(":");
    const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex);
    let value = separatorIndex === -1 ? "" : line.slice(separatorIndex + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    if (field === "id") {
      id = value;
    } else if (field === "event") {
      event = value || "message";
    } else if (field === "data") {
      dataLines.push(value);
    }
  }

  return { id, event, data: dataLines.join("\n") };
}

function formatSseFrame(event: string, data: string, id?: string | null): string {
  const idLine = id !== undefined && id !== null ? `id: ${id}\n` : "";
  return `${idLine}event: ${event}\ndata: ${data}\n\n`;
}

async function enrichWithTimeout(state: LiveDemoRunState): Promise<LiveDemoRunState> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      enrichRunStateWithPayload(state),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(
          () => reject(new EnrichmentTimeoutError()),
          ENRICH_TIMEOUT_MS
        );
      })
    ]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
  }
}

function createSseEnrichmentTransform(): {
  stream: TransformStream<string, string>;
  cleanup: () => void;
} {
  let buffer = "";
  let lastState: LiveDemoRunState | null = null;
  let lastEnrichedHash = "";
  let lastForwardAt = Date.now();
  let lastEventId: string | null = null;
  let syntheticCounter = 0;
  let interval: ReturnType<typeof setInterval> | null = null;
  let closed = false;
  let syntheticPending: Promise<void> | null = null;

  function warnEnrichmentFailure(context: string, error: unknown): void {
    const suffix = lastEventId ? ` after event ${lastEventId}` : "";
    console.warn(`[demo/events] ${context}${suffix}`, error);
  }

  function emitSyntheticIfChanged(
    controller: TransformStreamDefaultController<string>,
    context: string
  ): void {
    if (!lastState || closed || syntheticPending) {
      return;
    }

    syntheticPending = (async () => {
      try {
        const enriched = await enrichWithTimeout(lastState);
        if (closed) {
          return;
        }
        const enrichedHash = stableEnrichedHash(enriched);
        if (enrichedHash === lastEnrichedHash) {
          return;
        }
        syntheticCounter += 1;
        lastEnrichedHash = enrichedHash;
        lastForwardAt = Date.now();
        controller.enqueue(
          formatSseFrame("run_state", JSON.stringify(enriched), `synth-${syntheticCounter}`)
        );
      } catch (error) {
        warnEnrichmentFailure(context, error);
      }
    })().finally(() => {
      syntheticPending = null;
    });

    // Fire-and-forget: callers must not await this. Awaiting in the SSE
    // transform path would backpressure the stream by up to ENRICH_TIMEOUT_MS
    // per upstream frame, breaking the live experience under burst load.
  }

  async function handleFrame(
    frame: string,
    original: string,
    controller: TransformStreamDefaultController<string>
  ): Promise<void> {
    if (!frame.trim()) {
      controller.enqueue(original);
      return;
    }

    const parsed = parseSseFrame(frame);
    if (parsed.id !== null) {
      lastEventId = parsed.id;
    }

    if (parsed.event !== "run_state") {
      controller.enqueue(original);
      // Fire-and-forget; the syntheticPending guard debounces bursts.
      emitSyntheticIfChanged(controller, `Unable to synthesize state for ${parsed.event}`);
      return;
    }

    try {
      const state = JSON.parse(parsed.data) as LiveDemoRunState;
      lastState = state;
      const enriched = await enrichWithTimeout(state);
      lastEnrichedHash = stableEnrichedHash(enriched);
      lastForwardAt = Date.now();
      controller.enqueue(formatSseFrame("run_state", JSON.stringify(enriched), parsed.id));
    } catch (error) {
      warnEnrichmentFailure("Unable to enrich run_state", error);
      controller.enqueue(original);
    }
  }

  function cleanup(): void {
    closed = true;
    if (interval) {
      clearInterval(interval);
      interval = null;
    }
  }

  const stream = new TransformStream<string, string>({
    start(controller) {
      interval = setInterval(() => {
        if (!lastState || closed || Date.now() - lastForwardAt <= SYNTHETIC_IDLE_MS) {
          return;
        }
        emitSyntheticIfChanged(controller, "Unable to synthesize idle state");
      }, SYNTHETIC_INTERVAL_MS);
    },
    async transform(chunk, controller) {
      buffer += chunk;
      let match = /\r?\n\r?\n/.exec(buffer);
      while (match) {
        const boundaryEnd = match.index + match[0].length;
        const frame = buffer.slice(0, match.index);
        const original = buffer.slice(0, boundaryEnd);
        buffer = buffer.slice(boundaryEnd);
        await handleFrame(frame, original, controller);
        match = /\r?\n\r?\n/.exec(buffer);
      }
    },
    flush(controller) {
      cleanup();
      if (buffer) {
        controller.enqueue(buffer);
        buffer = "";
      }
    }
  });

  return { stream, cleanup };
}

function withCancelCleanup<T>(
  stream: ReadableStream<T>,
  cleanup: () => void
): ReadableStream<T> {
  let reader: ReadableStreamDefaultReader<T> | null = null;
  let finished = false;

  function finish(): void {
    if (finished) {
      return;
    }
    finished = true;
    cleanup();
  }

  return new ReadableStream<T>({
    start(controller) {
      reader = stream.getReader();
      void (async () => {
        try {
          while (reader && !finished) {
            const { done, value } = await reader.read();
            if (finished) return;
            if (done) {
              finish();
              controller.close();
              return;
            }
            controller.enqueue(value);
          }
        } catch (error) {
          if (finished) return;
          finish();
          try {
            controller.error(error);
          } catch {
            // Controller may already be closed by cancel(); ignore.
          }
        }
      })();
    },
    cancel(reason) {
      finish();
      const r = reader;
      reader = null;
      return r?.cancel(reason);
    }
  });
}

export async function GET(request: Request) {
  const projectId = new URL(request.url).searchParams.get("projectId");
  if (!projectId) {
    return NextResponse.json({ error: "projectId is required" }, { status: 400 });
  }

  try {
    const response = await fetch(
      `${backendBaseUrl()}/runs/${encodeURIComponent(projectId)}/events`,
      { cache: "no-store" }
    );
    if (!response.ok || !response.body) {
      return new NextResponse(await response.text(), { status: response.status });
    }
    const sseTransform = createSseEnrichmentTransform();
    const transformedBody = response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(sseTransform.stream)
      .pipeThrough(new TextEncoderStream());
    const body = withCancelCleanup(transformedBody, sseTransform.cleanup);

    return new Response(body, {
      status: response.status,
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "x-accel-buffering": "no"
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to open event stream";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
