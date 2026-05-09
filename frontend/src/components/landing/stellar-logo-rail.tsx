import { stellarLogos } from "../../lib/landing/stellar-tabs";

export function StellarLogoRail() {
  return (
    <div
      className="mt-24 flex flex-wrap items-center justify-center gap-8 text-sm font-medium text-gray-500 md:gap-12"
      style={{ opacity: 0, animationDelay: "0.8s" }}
    >
      {stellarLogos.map((logo) => {
        if (logo === "Nexera") {
          return (
            <div key={logo} className="flex items-center gap-3 animate-fade-in-up">
              <div className="grid grid-cols-2 gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
                <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
                <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
                <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
              </div>
              <span>{logo}</span>
            </div>
          );
        }

        if (logo === "LAURA COLE") {
          return (
            <div key={logo} className="flex items-center gap-3 animate-fade-in-up">
              <span className="flex h-8 w-8 items-center justify-center rounded-full border border-gray-300 text-xs text-black">
                LC
              </span>
              <span>{logo}</span>
            </div>
          );
        }

        if (logo === "vertex") {
          return (
            <div key={logo} className="flex items-center gap-3 lowercase animate-fade-in-up">
              <span>{logo}</span>
              <div className="flex gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
                <span className="h-1.5 w-1.5 rounded-full bg-gray-300" />
                <span className="h-1.5 w-1.5 rounded-full bg-gray-400" />
              </div>
            </div>
          );
        }

        if (logo === "M3") {
          return (
            <span key={logo} className="font-serif text-xl italic text-gray-500 animate-fade-in-up">
              {logo}
            </span>
          );
        }

        return (
          <span key={logo} className="animate-fade-in-up">
            {logo}
          </span>
        );
      })}
    </div>
  );
}
