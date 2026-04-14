import React, { useState } from "react";
import { useParams, useNavigate } from "react-router";
import { TopBar } from "../components/ui/TopBar";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/Button";
import { RadioGroup } from "../components/ui/RadioGroup";
import { TextArea } from "../components/ui/textarea";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

const mockBook = {
  id: 1,
  title: "Den Magiske Skogen",
  author: "Norsk Folkeeventyr",
  content: `
    <p class="mb-4">Det var en gang en gutt som het Espen Askeladd. Han vandret inn i Den Magiske Skogen for å finne Soria Moria slott.</p>
    <p class="mb-4">Skogen var ikke som andre skoger. Trærne hvisket til hverandre når vinden blåste, og blomstene lyste som små lamper i mørket. Det luktet av mose og mystikk.</p>
    <p class="mb-4">Plutselig hørte Espen en dyp brumming. Det var et stort troll som satt under en gammel steinbro! Trollet hadde tre hoder og en stor nese full av vorter.</p>
    <p class="mb-4">"Hvem er det som tramper på min bro?" ropte trollet med alle tre hodene samtidig. Stemmen ristet i bakken.</p>
    <p class="mb-4">"Det er bare jeg, Espen," sa gutten modig. Han visste at trollene sprakk hvis solen skinte på dem. Han måtte bare holde trollet opptatt til soloppgang.</p>
    <p class="mb-4">Espen begynte å fortelle gåter. "Hva er det som går og går, men aldri kommer til døra?" spurte han.</p>
    <p class="mb-4">Trollet klødde seg i alle tre hodene. Det elsket gåter, men var ikke veldig smart. De holdt på hele natten.</p>
    <p class="mb-4">Da de første solstrålene traff tretoppene, skrek trollet høyt. I det øyeblikket solen traff nesa hans, ble han til stein.</p>
    <p>Espen smilte, klappet steintrollet på nesa, og gikk plystrende videre inn i skogen. Eventyret hadde så vidt begynt.</p>
  `,
  image:
    "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmb3Jlc3QlMjBtYWdpYyUyMHRyZWVzfGVufDF8fHx8MTc3MjI3ODUyNHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
};

type QuestionVariant =
  | "collapsed"
  | "mc"
  | "short"
  | "tf"
  | "rating"
  | "submit";

export function Reading() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [variant, setVariant] =
    useState<QuestionVariant>("collapsed");

  // State for answers
  const [mcAnswer, setMcAnswer] = useState<string>("");
  const [shortAnswer, setShortAnswer] = useState("");
  const [tfAnswers, setTfAnswers] = useState<{
    [key: number]: string;
  }>({});
  const [rating, setRating] = useState<number | null>(null);

  const handleNext = () => {
    if (variant === "collapsed") setVariant("mc");
    else if (variant === "mc") setVariant("short");
    else if (variant === "short") setVariant("tf");
    else if (variant === "tf") setVariant("rating");
    else if (variant === "rating") setVariant("submit");
  };

  const handlePrev = () => {
    if (variant === "mc") setVariant("collapsed");
    else if (variant === "short") setVariant("mc");
    else if (variant === "tf") setVariant("short");
    else if (variant === "rating") setVariant("tf");
    else if (variant === "submit") setVariant("rating");
  };

  const handleSubmit = () => {
    navigate("/completion");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar />

      <main className="flex-1 container mx-auto px-4 py-8 max-w-[1440px]">
        <div className="grid lg:grid-cols-12 gap-8 h-full">
          {/* Left Column - Reading Content */}
          <div className="lg:col-span-7 pb-20">
            <Card className="p-8 h-full border border-[#e0ddd5]">
              <h1 className="text-3xl font-bold text-[#2d3142] mb-6">
                {mockBook.title}
              </h1>

              <div className="w-full h-64 md:h-80 rounded-[24px] overflow-hidden shadow-md mb-6">
                <img
                  src={mockBook.image}
                  alt={mockBook.title}
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-sm text-[#5d6875] italic mb-6">
                Illustrasjon av Unsplash Artister
              </p>

              <div
                className="prose prose-lg prose-slate max-w-none text-[#2d3142] leading-relaxed"
                dangerouslySetInnerHTML={{
                  __html: mockBook.content,
                }}
              />
            </Card>
          </div>

          {/* Right Column - Question Panel */}
          <div className="lg:col-span-5 relative">
            <div className="sticky top-24">
              <Card className="bg-white border border-[#e0ddd5] min-h-[300px] flex flex-col relative overflow-hidden">
                <AnimatePresence mode="wait">
                  {variant === "collapsed" && (
                    <motion.div
                      key="collapsed"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center p-8 space-y-4 text-center"
                    >
                      <Button onClick={handleNext}>
                        Vis spørsmål <ArrowRight size={18} />
                      </Button>
                    </motion.div>
                  )}

                  {variant === "mc" && (
                    <QuestionContainer
                      key="mc"
                      step={1}
                      totalSteps={4}
                      onNext={handleNext}
                      onPrev={handlePrev}
                      showPrev={false}
                    >
                      <div className="space-y-6">
                        <div className="flex items-start gap-3">
                          <Badge number={1} />
                          <h3 className="text-xl font-semibold text-[#2d3142] mt-1">
                            Hvem møtte Espen under broen?
                          </h3>
                        </div>

                        <div className="pl-12">
                          <RadioGroup
                            name="q1"
                            value={mcAnswer}
                            onChange={setMcAnswer}
                            options={[
                              {
                                value: "En heks",
                                label: "En heks",
                              },
                              {
                                value: "Et troll med tre hoder",
                                label: "Et troll med tre hoder",
                              },
                              {
                                value: "En snakkende rev",
                                label: "En snakkende rev",
                              },
                              {
                                value: "Kongen av skogen",
                                label: "Kongen av skogen",
                              },
                            ]}
                          />
                        </div>
                      </div>
                    </QuestionContainer>
                  )}

                  {variant === "short" && (
                    <QuestionContainer
                      key="short"
                      step={2}
                      totalSteps={4}
                      onNext={handleNext}
                      onPrev={handlePrev}
                    >
                      <div className="space-y-6">
                        <div className="flex items-start gap-3">
                          <Badge number={2} />
                          <h3 className="text-xl font-semibold text-[#2d3142] mt-1">
                            Hvorfor fortalte Espen gåter til
                            trollet?
                          </h3>
                        </div>

                        <div className="pl-12 space-y-2">
                          <TextArea
                            placeholder="Skriv svaret ditt her..."
                            maxLength={190}
                            value={shortAnswer}
                            onChange={(e) =>
                              setShortAnswer(e.target.value)
                            }
                            className="h-32"
                          />
                          <div className="text-right text-xs text-[#5d6875]">
                            {shortAnswer.length} / 190 tegn
                          </div>
                        </div>
                      </div>
                    </QuestionContainer>
                  )}

                  {variant === "tf" && (
                    <QuestionContainer
                      key="tf"
                      step={3}
                      totalSteps={4}
                      onNext={handleNext}
                      onPrev={handlePrev}
                    >
                      <div className="space-y-6">
                        <div className="flex items-start gap-3">
                          <Badge number={3} />
                          <h3 className="text-xl font-semibold text-[#2d3142] mt-1">
                            Sant eller Usant?
                          </h3>
                        </div>

                        <div className="pl-12 space-y-4">
                          {[
                            {
                              id: 1,
                              text: "Trollet ble til stein da solen sto opp.",
                            },
                            {
                              id: 2,
                              text: "Espen var redd og løp hjem.",
                            },
                            {
                              id: 3,
                              text: "Skogen var helt mørk uten lys.",
                            },
                          ].map((stmt) => (
                            <div
                              key={stmt.id}
                              className="bg-white p-4 rounded-[16px] border border-[#e0ddd5]"
                            >
                              <p className="mb-3 text-[#2d3142] font-medium">
                                {stmt.text}
                              </p>
                              <div className="flex gap-6">
                                <label className="flex items-center gap-2 cursor-pointer">
                                  <input
                                    type="radio"
                                    name={`tf-${stmt.id}`}
                                    className="w-4 h-4 accent-[#4ecdc4]"
                                    checked={
                                      tfAnswers[stmt.id] ===
                                      "true"
                                    }
                                    onChange={() =>
                                      setTfAnswers({
                                        ...tfAnswers,
                                        [stmt.id]: "true",
                                      })
                                    }
                                  />
                                  <span>Sant</span>
                                </label>
                                <label className="flex items-center gap-2 cursor-pointer">
                                  <input
                                    type="radio"
                                    name={`tf-${stmt.id}`}
                                    className="w-4 h-4 accent-[#4ecdc4]"
                                    checked={
                                      tfAnswers[stmt.id] ===
                                      "false"
                                    }
                                    onChange={() =>
                                      setTfAnswers({
                                        ...tfAnswers,
                                        [stmt.id]: "false",
                                      })
                                    }
                                  />
                                  <span>Usant</span>
                                </label>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </QuestionContainer>
                  )}

                  {variant === "rating" && (
                    <QuestionContainer
                      key="rating"
                      step={4}
                      totalSteps={4}
                      onNext={handleNext}
                      onPrev={handlePrev}
                    >
                      <div className="space-y-6">
                        <div className="flex items-start gap-3">
                          <Badge number={4} />
                          <h3 className="text-xl font-semibold text-[#2d3142] mt-1">
                            Hvor godt likte du teksten?
                          </h3>
                        </div>

                        <div className="pl-12 flex flex-row flex-wrap gap-6 items-end">
                          {[
                            {
                              val: 5,
                              label: "Veldig godt",
                              size: "w-16 h-16",
                            },
                            {
                              val: 4,
                              label: "Godt",
                              size: "w-14 h-14",
                            },
                            {
                              val: 3,
                              label: "Middels",
                              size: "w-12 h-12",
                            },
                            {
                              val: 2,
                              label: "Lite",
                              size: "w-10 h-10",
                            },
                            {
                              val: 1,
                              label: "Veldig lite",
                              size: "w-8 h-8",
                            },
                          ].map((opt) => (
                            <div
                              key={opt.val}
                              className="flex flex-col items-center gap-2 group cursor-pointer"
                              onClick={() => setRating(opt.val)}
                            >
                              <div
                                className={`
                                  rounded-full border-2 flex items-center justify-center transition-all duration-300
                                  ${rating === opt.val ? "border-[#4ecdc4] bg-[#4ecdc4]/10" : "border-[#e0ddd5] group-hover:border-[#95b8a2]"}
                                  ${opt.size}
                                `}
                              >
                                {rating === opt.val && (
                                  <div className="w-[50%] h-[50%] bg-[#4ecdc4] rounded-full" />
                                )}
                              </div>
                              <span
                                className={`font-medium text-lg ${rating === opt.val ? "text-[#4ecdc4]" : "text-[#5d6875]"}`}
                              >
                                {opt.label}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </QuestionContainer>
                  )}

                  {variant === "submit" && (
                    <motion.div
                      key="submit"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="flex-1 flex flex-col items-center justify-center p-8 space-y-6 text-center"
                    >
                      <div className="w-16 h-16 bg-[#e07a5f]/10 rounded-full flex items-center justify-center mb-2">
                        <CheckCircle2
                          size={32}
                          className="text-[#e07a5f]"
                        />
                      </div>
                      <h3 className="text-xl font-semibold text-[#2d3142]">
                        Nå har du gått gjennom alle spørsmålene,
                        vil du sende inn?
                      </h3>

                      <div className="flex flex-col gap-4 w-full max-w-xs">
                        <Button
                          variant="completion"
                          onClick={handleSubmit}
                        >
                          Send inn
                        </Button>
                        <button
                          onClick={handlePrev}
                          className="text-[#5d6875] hover:text-[#4ecdc4] flex items-center justify-center gap-2 transition-colors"
                        >
                          <ArrowLeft size={16} /> Tilbake til
                          spørsmålene
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function Badge({ number }: { number: number }) {
  return (
    <div className="w-8 h-8 rounded-full bg-[#2d3142] text-white flex items-center justify-center font-bold text-sm flex-shrink-0">
      {number}
    </div>
  );
}

function QuestionContainer({
  children,
  step,
  totalSteps,
  onNext,
  onPrev,
  showPrev = true,
}: {
  children: React.ReactNode;
  step: number;
  totalSteps: number;
  onNext: () => void;
  onPrev: () => void;
  showPrev?: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="flex flex-col h-full p-8"
    >
      <div className="flex-1">{children}</div>

      <div className="mt-8 pt-6 border-t border-[#e0ddd5] grid grid-cols-3 items-center">
        <div className="justify-self-start">
          {showPrev && (
            <Button variant="neutral" onClick={onPrev}>
              Forrige
            </Button>
          )}
        </div>

        <span className="text-[#5d6875] font-medium text-sm justify-self-center">
          {step} av {totalSteps}
        </span>

        <div className="justify-self-end">
          <Button onClick={onNext}>Neste</Button>
        </div>
      </div>
    </motion.div>
  );
}