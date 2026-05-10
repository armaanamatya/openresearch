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
        <span className="block">Reproduce Smarter.</span>
        <span className="block bg-gradient-to-r from-black via-gray-500 to-gray-400 bg-clip-text text-transparent">
          Verify Faster.
        </span>
      </h1>

      <p
        className="mx-auto mb-7 max-w-2xl text-lg text-gray-600 md:text-xl animate-fade-in-up"
        style={{ opacity: 0, animationDelay: "0.3s" }}
      >
        Your multi-agent system is reading the paper, building the environment, and
        verifying the results - independently.
      </p>

      <a
        className="mb-9 inline-flex items-center justify-center rounded-full bg-black px-8 py-3 text-base font-medium text-white transition-colors hover:bg-gray-800 animate-fade-in-up"
        href="/lab"
        style={{ opacity: 0, animationDelay: "0.4s" }}
      >
        Start Now
      </a>

      <StellarTabStage />
      <StellarLogoRail />

      {children}
    </section>
  );
}
