import { Star } from "lucide-react";

import { StellarTabStage } from "./stellar-tab-stage";

type StellarHeroProps = {
  children?: React.ReactNode;
};

export function StellarHero({ children }: StellarHeroProps) {
  return (
    <section className="mx-auto max-w-7xl px-6 pb-32 pt-24 text-center">
      <div
        className="mb-8 inline-flex items-center gap-2 animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.2s" }}
      >
        <div className="flex h-6 w-6 items-center justify-center rounded border border-gray-300">
          <Star className="h-3.5 w-3.5 fill-black text-black" />
        </div>
        <span className="text-sm font-medium text-black">4.9 rating from 18.3K+ users</span>
      </div>

      <h1
        className="mb-5 text-6xl font-normal leading-[1.1] tracking-tight text-black md:text-7xl lg:text-[80px] animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.3s" }}
      >
        <span className="block">Work Smarter. Move Faster.</span>
        <span className="block bg-gradient-to-r from-black via-gray-500 to-gray-400 bg-clip-text text-transparent">
          AI Powers You Up.
        </span>
      </h1>

      <p
        className="mx-auto mb-8 max-w-2xl text-lg text-gray-600 md:text-xl animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.4s" }}
      >
        Intelligent automation syncs with the tools you love to streamline tasks, boost
        output, and save time.
      </p>

      <button
        className="mb-12 rounded-full bg-black px-8 py-3 text-base font-medium text-white transition-colors hover:bg-gray-800 animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.5s" }}
        type="button"
      >
        Begin Free Trial
      </button>

      <StellarTabStage />

      {children}
    </section>
  );
}
