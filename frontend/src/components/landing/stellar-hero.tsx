import { StellarLogoRail } from "./stellar-logo-rail";
import { StellarTabStage } from "./stellar-tab-stage";

type StellarHeroProps = {
  children?: React.ReactNode;
};

export function StellarHero({ children }: StellarHeroProps) {
  return (
    <section className="mx-auto max-w-7xl px-6 pb-24 pt-12 text-center md:pt-14 lg:pt-16">
      <h1
        className="mb-4 text-6xl font-normal leading-[1.1] tracking-tight text-black md:text-7xl lg:text-[80px] animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.2s" }}
      >
        <span className="block">Work Smarter. Move Faster.</span>
        <span className="block bg-gradient-to-r from-black via-gray-500 to-gray-400 bg-clip-text text-transparent">
          AI Powers You Up.
        </span>
      </h1>

      <p
        className="mx-auto mb-7 max-w-2xl text-lg text-gray-600 md:text-xl animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.3s" }}
      >
        Intelligent automation syncs with the tools you love to streamline tasks, boost
        output, and save time.
      </p>

      <button
        className="mb-9 rounded-full bg-black px-8 py-3 text-base font-medium text-white transition-colors hover:bg-gray-800 animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.4s" }}
        type="button"
      >
        Begin Free Trial
      </button>

      <StellarTabStage />
      <StellarLogoRail />

      {children}
    </section>
  );
}
