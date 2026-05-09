"use client";

import { useEffect, useState } from "react";
import { Check, CheckCircle2 } from "lucide-react";

import {
  stellarOverlays,
  stellarTabs,
  stellarVideoSource,
  type StellarTabId
} from "../../lib/landing/stellar-tabs";

const cycleOrder: StellarTabId[] = ["analyse", "train", "testing", "deploy"];

export function StellarTabStage() {
  const [activeTab, setActiveTab] = useState<StellarTabId>("analyse");
  const activeOverlay = stellarOverlays[activeTab];

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setActiveTab((current) => {
        const currentIndex = cycleOrder.indexOf(current);
        const nextIndex = (currentIndex + 1) % cycleOrder.length;
        return cycleOrder[nextIndex];
      });
    }, 4000);

    return () => window.clearInterval(intervalId);
  }, []);

  return (
    <div
      className="animate-fade-in-up"
      style={{ opacity: 0, animationDelay: "0.6s" }}
    >
      <div className="mx-auto mb-8 max-w-3xl rounded-lg bg-gray-100 p-1">
        <div className="grid grid-cols-2 gap-1 md:hidden">
          {stellarTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = tab.id === activeTab;

            return (
              <button
                key={tab.id}
                className={`flex items-center justify-center gap-2 rounded-md px-4 py-3 text-sm font-medium transition-all ${
                  isActive ? "bg-white text-black shadow-sm" : "text-gray-600"
                }`}
                onClick={() => setActiveTab(tab.id)}
                aria-pressed={isActive}
                type="button"
              >
                <Icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="hidden items-center justify-center md:flex">
          {stellarTabs.map((tab, index) => {
            const Icon = tab.icon;
            const isActive = tab.id === activeTab;

            return (
              <div key={tab.id} className="flex items-center">
                <button
                  className={`flex items-center gap-2 rounded-md px-6 py-3 text-sm font-medium transition-all ${
                    isActive ? "bg-white text-black shadow-sm" : "text-gray-600"
                  }`}
                  onClick={() => setActiveTab(tab.id)}
                  aria-pressed={isActive}
                  type="button"
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.label}</span>
                </button>
                {index < stellarTabs.length - 1 ? (
                  <div className="mx-2 h-5 w-px bg-gray-300" />
                ) : null}
              </div>
            );
          })}
        </div>
      </div>

      <div
        className="relative h-[400px] overflow-hidden rounded-3xl md:h-[500px] animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.7s" }}
      >
        <video
          autoPlay
          className="h-full w-full object-cover"
          data-testid="stellar-video"
          loop
          muted
          playsInline
          src={stellarVideoSource}
        />
        <div className="absolute inset-0 bg-black/10" />
        <div className="animate-fade-in-overlay absolute inset-0">
          <div className="animate-slide-up-overlay absolute left-1/2 top-1/2 w-[88%] max-w-md rounded-3xl border border-white/60 bg-white/95 p-6 text-left shadow-2xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="mb-1 text-xs font-semibold uppercase tracking-[0.2em] text-gray-500">
                  {activeOverlay.eyebrow}
                </p>
                <h2 className="text-2xl font-semibold text-black">{activeOverlay.title}</h2>
              </div>
              <span className={`h-3 w-3 rounded-full ${activeOverlay.accentClass}`} />
            </div>

            <p className="mb-5 text-sm leading-6 text-gray-600">{activeOverlay.description}</p>

            {typeof activeOverlay.progress === "number" ? (
              <div className="mb-5">
                <div className="mb-2 h-2 rounded-full bg-gray-200">
                  <div
                    className={`h-2 rounded-full ${activeOverlay.accentClass}`}
                    style={{ width: `${activeOverlay.progress}%` }}
                  />
                </div>
                <p className="text-sm font-medium text-gray-700">{activeOverlay.progress}% complete</p>
              </div>
            ) : null}

            {activeOverlay.steps ? (
              <div className="grid gap-3 sm:grid-cols-2">
                {activeOverlay.steps.map((step, index) => (
                  <div
                    key={step}
                    className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700"
                  >
                    <span className="mb-1 block text-xs font-semibold uppercase tracking-[0.18em] text-gray-400">
                      Step {index + 1}
                    </span>
                    {step}
                  </div>
                ))}
              </div>
            ) : null}

            {activeOverlay.metrics ? (
              <div className="grid gap-3 sm:grid-cols-2">
                {activeOverlay.metrics.map((metric) => (
                  <div
                    key={metric.label}
                    className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3"
                  >
                    <span className="block text-xs uppercase tracking-[0.18em] text-gray-400">
                      {metric.label}
                    </span>
                    <span className="mt-2 block text-lg font-semibold text-black">{metric.value}</span>
                  </div>
                ))}
              </div>
            ) : null}

            {activeOverlay.successLabel ? (
              <div className="rounded-2xl border border-emerald-100 bg-emerald-50 px-4 py-4">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                  <div>
                    <p className="text-sm font-medium text-emerald-700">Validation complete</p>
                    <p className="text-lg font-semibold text-emerald-900">
                      {activeOverlay.successLabel}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            {activeOverlay.checklist ? (
              <div className="space-y-3">
                {activeOverlay.checklist.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-700"
                  >
                    <span>{item.label}</span>
                    <span
                      className={`flex h-6 w-6 items-center justify-center rounded-full ${
                        item.complete ? "bg-black text-white" : "border border-gray-300 text-gray-400"
                      }`}
                    >
                      {item.complete ? <Check className="h-4 w-4" /> : <span className="h-2 w-2 rounded-full bg-current" />}
                    </span>
                  </div>
                ))}
                {activeOverlay.ctaLabel ? (
                  <button
                    className="w-full rounded-full bg-black px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-gray-800"
                    type="button"
                  >
                    {activeOverlay.ctaLabel}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
