import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { motion } from "motion/react";
import { Check } from "lucide-react";
import { Button } from "../components/ui/Button";
import { getBroadTopics, saveInterests, type BroadTopic } from "../lib/session";

export function Interests() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [topics, setTopics] = useState<BroadTopic[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    const loadTopics = async () => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const fetchedTopics = await getBroadTopics();
        if (!active) return;
        setTopics(fetchedTopics);
      } catch (error) {
        if (!active) return;
        const message = error instanceof Error
          ? error.message
          : "Kunne ikke laste interesser.";
        setLoadError(message);
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    };

    void loadTopics();

    return () => {
      active = false;
    };
  }, []);

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

  const handleContinue = async () => {
    if (selected.size < 3 || isSaving) return;

    setSaveError(null);
    setIsSaving(true);

    try {
      await saveInterests(Array.from(selected));
      navigate("/dashboard");
    } catch (error) {
      const message = error instanceof Error
        ? error.message
        : "Kunne ikke lagre interesser.";
      setSaveError(message);
    } finally {
      setIsSaving(false);
    }
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
            Velg minst tre temaer - vi finner tekster du vil elske!
          </p>
        </div>

        {/* Genre grid */}
        {isLoading && (
          <div className="text-center text-[#5d6875]">Laster interesser...</div>
        )}

        {!isLoading && loadError && (
          <div className="text-center text-red-600">{loadError}</div>
        )}

        {!isLoading && !loadError && (
          <div className="flex flex-wrap justify-center gap-4">
            {topics.map((topic, i) => {
              const isSelected = selected.has(topic.name);
              return (
                <motion.button
                  key={topic.id}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.04, duration: 0.3 }}
                  onClick={() => toggle(topic.name)}
                  className={`
                    relative flex items-center justify-center w-[148px] sm:w-[168px] min-h-[88px] p-4 rounded-[24px] border-2 transition-all duration-200 text-center cursor-pointer
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

                  <span className="font-semibold text-[#2d3142]">
                    {topic.name}
                  </span>
                </motion.button>
              );
            })}
          </div>
        )}

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
              disabled={selected.size < 3 || isLoading || isSaving || !!loadError}
            >
              {isSaving ? "Lagrer..." : "Fortsett"}
              {selected.size > 0 && (
                <span className="ml-2 bg-white/30 text-white text-xs font-bold px-2 py-0.5 rounded-full">
                  {selected.size}
                </span>
              )}
            </Button>
          </motion.div>
          {saveError && (
            <p className="text-sm text-red-600 text-center">{saveError}</p>
          )}
        </div>

      </div>
    </div>
  );
}