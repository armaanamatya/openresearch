import { StellarHero } from "../components/landing/stellar-hero";
import { StellarNavigation } from "../components/landing/stellar-navigation";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-white">
      <StellarNavigation />
      <StellarHero />
    </main>
  );
}
