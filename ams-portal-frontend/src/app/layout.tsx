import "./globals.css";
import type { Metadata } from 'next';
import { IBM_Plex_Sans } from 'next/font/google';
import { Providers } from './providers';

// Load the IBM Plex Sans font from Google Fonts
const ibm = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],  // Include multiple font weights
});

// Define metadata for the application
export const metadata: Metadata = {
  title: 'NextGen AMS Knowledge Assistant',  // Kept title unchanged
  description: 'NextGen AMS Knowledge Assistant - Self-Serve Portal',  // Updated description
};

// Define the RootLayout component for the application
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Set metadata for the application */}
        <title>NextGen AMS Knowledge Assistant</title>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="description" content="NextGen AMS Knowledge Assistant - Self-Serve Portal" />
      </head>
      <body className={ibm.className}>
        <Providers>
          {/* Render the child components inside Providers */}
          {children}
        </Providers>
      </body>
    </html>
  );
}
