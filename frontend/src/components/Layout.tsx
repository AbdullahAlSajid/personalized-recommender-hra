import { Outlet } from "react-router";

export function Layout() {
  return (
    <div className="min-h-screen w-full relative overflow-x-hidden font-sans text-[#2d3142] bg-[#faf8f5]">
      {/* Background Blobs */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div
          className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] rounded-full blur-3xl opacity-20"
          style={{ background: 'radial-gradient(circle, #4ecdc4, #95b8a2)' }}
        />
        <div
          className="absolute bottom-[-10%] right-[-10%] w-[50vw] h-[50vw] rounded-full blur-3xl opacity-20"
          style={{ background: 'radial-gradient(circle, #ff6b6b, #e07a5f)' }}
        />
      </div>

      <div className="relative z-10">
        <Outlet />
      </div>
    </div>
  );
}
