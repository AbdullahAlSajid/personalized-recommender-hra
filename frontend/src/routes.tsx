import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Passcode } from "./pages/Passcode";
import { Consent } from "./pages/Consent";
import { Login } from "./pages/Login";
import { Interests } from "./pages/Interests";
import { Dashboard } from "./pages/Dashboard";
import { Reading } from "./pages/Reading";
import { Completion } from "./pages/Completion";
import { SessionFeedback } from "./pages/SessionFeedback";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Passcode },
      { path: "consent", Component: Consent },
      { path: "login", Component: Login },
      { path: "interests", Component: Interests },
      { path: "dashboard", Component: Dashboard },
      { path: "reading/:id", Component: Reading },
      { path: "completion", Component: Completion },
      { path: "session-feedback", Component: SessionFeedback },
    ],
  },
]);