import React, { useState } from "react";
import { useNavigate } from "react-router";
import { motion } from "motion/react";
import { Check } from "lucide-react";
import { Button } from "../components/ui/Button";

const genres = [
  { id: "eventyr",    label: "Eventyr",    emoji: "🏰", description: "Riddere, drager og skatter" },
  { id: "dyr",        label: "Dyr",        emoji: "🐾", description: "Ville og tamme dyr" },
  { id: "magi",       label: "Magi",       emoji: "✨", description: "Trolldom og tryllekunst" },
  { id: "sport",      label: "Sport",      emoji: "⚽", description: "Fotball, svømming og mer" },
  { id: "natur",      label: "Naturen",    emoji: "🌿", description: "Fjell, hav og skog" },
  { id: "vitenskap",  label: "Vitenskap",  emoji: "🔬", description: "Eksperimenter og oppdagelser" },
  { id: "humor",      label: "Humor",      emoji: "😂", description: "Morsomme historier" },
  { id: "mysterium",  label: "Mysterium",  emoji: "🔍", description: "Gåter og hemmeligheter" },
  { id: "romfart",    label: "Romfart",    emoji: "🚀", description: "Planeter og astronauter" },
  { id: "vennskap",   label: "Vennskap",   emoji: "🤝", description: "Historier om gode venner" },
  { id: "historie",   label: "Historie",   emoji: "📜", description: "Vikinger og gamle tider" },
  { id: "musikk",     label: "Musikk",     emoji: "🎵", description: "Sanger og instrumenter" },
];

export function Interests() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleContinue = () => {
    // Persist to localStorage for use in recommendations
    localStorage.setItem("recsys_interests", JSON.stringify([...selected]));
    navigate("/dashboard");
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12 relative z-10">
      <div className="w-full max-w-3xl flex flex-col gap-10">

        {/* Header */}
        <div className="text-center space-y-3">
          <h1 className="text-4xl font-bold text-[#2d3142]">
            Hva liker du å lese om?
          </h1>
          <p className="text-[#5d6875] text-lg">
            Velg minst tre sjangere — vi finner tekster du vil elske!
          </p>
        </div>

        {/* Genre grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
          {genres.map((genre, i) => {
            const isSelected = selected.has(genre.id);
            return (
              <motion.button
                key={genre.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.04, duration: 0.3 }}
                onClick={() => toggle(genre.id)}
                className={`
                  relative flex flex-col items-center gap-2 p-5 rounded-[24px] border-2 transition-all duration-200 text-center cursor-pointer
                  ${isSelected
                    ? "border-[#4ecdc4] bg-[#4ecdc4]/10 shadow-[0_8px_30px_rgba(78,205,196,0.2)]"
                    : "border-[#e0ddd5] bg-white hover:border-[#95b8a2] hover:shadow-[0_8px_30px_rgba(0,0,0,0.08)]"
                  }
                `}
              >
                {/* Check badge */}
                {isSelected && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute top-3 right-3 w-5 h-5 rounded-full bg-[#4ecdc4] flex items-center justify-center"
                  >
                    <Check size={11} strokeWidth={3} className="text-white" />
                  </motion.div>
                )}

                <span className="text-4xl">{genre.emoji}</span>
                <span className={`font-semibold ${isSelected ? "text-[#2d3142]" : "text-[#2d3142]"}`}>
                  {genre.label}
                </span>
                <span className="text-xs text-[#5d6875] leading-snug">
                  {genre.description}
                </span>
              </motion.button>
            );
          })}
        </div>

        {/* Footer */}
        <div className="flex flex-col items-center gap-4">
          <motion.div
            animate={{ opacity: selected.size < 3 ? 0.4 : 1 }}
            transition={{ duration: 0.2 }}
            className="w-full max-w-xs"
          >
            <Button
              className="w-full"
              onClick={handleContinue}
              disabled={selected.size < 3}
            >
              Fortsett
              {selected.size > 0 && (
                <span className="ml-2 bg-white/30 text-white text-xs font-bold px-2 py-0.5 rounded-full">
                  {selected.size}
                </span>
              )}
            </Button>
          </motion.div>
        </div>

      </div>
    </div>
  );
}