import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI Trading Lab",
  description: "Collaborative AI Funds - Multiple AI models debating and trading together",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="min-h-screen bg-background">
          <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
            <div className="container mx-auto px-4 py-3">
              <div className="flex items-center justify-between">
                <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
                  <span className="text-2xl">🧪</span>
                  <div>
                    <h1 className="text-xl font-bold">AI Trading Lab</h1>
                    <p className="text-xs text-muted-foreground">
                      Collaborative AI Funds
                    </p>
                  </div>
                </Link>
                <nav className="flex items-center gap-1">
                  <NavLink href="/">Dashboard</NavLink>
                  <NavLink href="/search">Search</NavLink>
                  <NavLink href="/funds">Funds</NavLink>
                  <NavLink href="/research">Research</NavLink>
                  <NavLink href="/signals">Signals</NavLink>
                </nav>
              </div>
            </div>
          </header>
          <main className="container mx-auto px-4 py-6">
            {children}
          </main>
          <footer className="border-t py-4 mt-8">
            <div className="container mx-auto px-4 text-center text-xs text-muted-foreground">
              AI Trading Lab - Where multiple AI models collaborate via structured debate
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors"
    >
      {children}
    </Link>
  );
}
