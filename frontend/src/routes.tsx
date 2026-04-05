import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Login } from "./pages/Login";
import { Interests } from "./pages/Interests";
import { Dashboard } from "./pages/Dashboard";
import { Reading } from "./pages/Reading";
import { Completion } from "./pages/Completion";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Layout,
    children: [
      { index: true, Component: Login },
      { path: "interests", Component: Interests },
      { path: "dashboard", Component: Dashboard },
      { path: "reading/:id", Component: Reading },
      { path: "completion", Component: Completion },
    ],
  },
]);