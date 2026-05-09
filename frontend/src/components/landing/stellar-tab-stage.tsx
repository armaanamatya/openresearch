"use client";

import { useEffect, useState } from "react";

import { stellarTabs, stellarVideoSource, type StellarTabId } from "../../lib/landing/stellar-tabs";

const cycleOrder: StellarTabId[] = ["analyse", "train", "testing", "deploy"];

export function StellarTabStage() {
  const [activeTab, setActiveTab] = useState<StellarTabId>("analyse");

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
      </div>
    </div>
  );
}
