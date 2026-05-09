import { ChevronDown, Star } from "lucide-react";

export function StellarNavigation() {
  return (
    <nav
      className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 animate-fade-in-up"
      style={{ opacity: 0, animationDelay: "0.1s" }}
    >
      <div className="flex items-center gap-2">
        <Star className="h-5 w-5 fill-black text-black" />
        <span className="text-lg font-semibold text-black">Stellar.ai</span>
      </div>

      <div className="hidden items-center gap-8 md:flex">
        <button className="flex items-center gap-1 text-sm text-gray-700 transition-colors hover:text-black" type="button">
          <span>Solutions</span>
          <ChevronDown className="h-4 w-4" />
        </button>
        <button className="flex items-center gap-1 text-sm text-gray-700 transition-colors hover:text-black" type="button">
          <span>For Teams</span>
          <ChevronDown className="h-4 w-4" />
        </button>
        <a className="text-sm text-gray-700 transition-colors hover:text-black" href="#about">
          About Us
        </a>
        <a className="text-sm text-gray-700 transition-colors hover:text-black" href="#learn">
          Learn Hub
        </a>
      </div>

      <div className="flex items-center gap-4">
        <a className="text-sm text-gray-700 transition-colors hover:text-black" href="#login">
          Login
        </a>
        <button
          className="rounded-full bg-black px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-gray-800"
          type="button"
        >
          Get started free
        </button>
      </div>
    </nav>
  );
}
