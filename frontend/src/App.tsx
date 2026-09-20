import { BrowserRouter, Route, Routes } from "react-router";
import { SpeedInsights } from "@vercel/speed-insights/react";

import RootLayout from "./layouts/RootLayout";
import AnalyzerPage from "./pages/AnalyzerPage";
import HelpPage from "./pages/HelpPage";
import LoginPage from "./pages/LoginPage";
import NotFoundPage from "./pages/NotFoundPage";
import PrivacyPage from "./pages/PrivacyPage";
import TermsPage from "./pages/TermsPage";

/**
 * Declarative `<Routes>`/`<Route>` JSX, not `createBrowserRouter`'s config
 * object: the same "keep the wiring visible" reasoning CONVENTIONS.md
 * applies to the backend's hand-written LangGraph graph and this frontend's
 * hand-written ESLint flat config — the route tree should read like a route
 * tree, not be reconstructed from a data structure.
 *
 * "/" is public — it shows the analyzer shell (upload zone, empty
 * chart/table, chat) to anyone. Login is only required per-action (upload,
 * chat), via the shared modal `requireAuth()` opens — see `auth/authModal.ts`
 * and RAPPORT.md. No route here is gated by a redirect anymore.
 */
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<RootLayout />}>
          <Route index element={<AnalyzerPage />} />
          <Route path="login" element={<LoginPage />} />
          <Route path="aide" element={<HelpPage />} />
          <Route path="confidentialite" element={<PrivacyPage />} />
          <Route path="conditions-utilisation" element={<TermsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
      <SpeedInsights />
    </BrowserRouter>
  );
}
