import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { TopBar } from "../components/ui/TopBar";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/Button";
import { RadioGroup } from "../components/ui/RadioGroup";
import {
  BACKEND_URL,
  endSession,
  getSessionFeedbackTexts,
  submitSessionFeedback,
  type SessionFeedbackText,
} from "../lib/session";

type FeedbackStepKey = "q1" | "q2" | "favorite" | "favoriteWhy" | "submit";
type QuestionStepKey = Exclude<FeedbackStepKey, "submit">;

const FAVORITE_NONE = "__none__";

export function SessionFeedback() {
  const navigate = useNavigate();

  const [textsLoading, setTextsLoading] = useState(false);
  const [textsError, setTextsError] = useState<string | null>(null);
  const [texts, setTexts] = useState<SessionFeedbackText[]>([]);

  const [q1, setQ1] = useState("");
  const [q2, setQ2] = useState("");

  const [favoriteTextId, setFavoriteTextId] = useState("");
  const [favoriteWhy, setFavoriteWhy] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    setTextsError(null);
    setTextsLoading(true);
    getSessionFeedbackTexts()
      .then((data) => setTexts(data))
      .catch(() => setTextsError("Kunne ikke hente tekstene fra økten."))
      .finally(() => setTextsLoading(false));
  }, []);

  const favoriteOptions = useMemo(
    () =>
      texts.map((t) => ({
        text_id: t.text_id,
        title: t.title,
        first_image_url: t.first_image_url ?? null,
      })),
    [texts]
  );

  const canSubmit =
    q1 !== "" &&
    q2 !== "" &&
    (favoriteOptions.length === 0 ||
      (favoriteTextId !== "" &&
        (favoriteTextId === FAVORITE_NONE || favoriteWhy !== "")));

  const steps: FeedbackStepKey[] = useMemo(() => {
    const out: FeedbackStepKey[] = ["q1", "q2", "favorite"];
    if (
      favoriteOptions.length > 0 &&
      favoriteTextId !== "" &&
      favoriteTextId !== FAVORITE_NONE
    ) {
      out.push("favoriteWhy");
    }
    out.push("submit");
    return out;
  }, [favoriteOptions.length, favoriteTextId]);

  useEffect(() => {
    setStepIndex((current) => Math.min(current, Math.max(0, steps.length - 1)));
  }, [steps.length]);

  const currentStep: FeedbackStepKey = steps[stepIndex] ?? "q1";

  const questionSteps = steps.filter(
    (k): k is QuestionStepKey => k !== "submit"
  );
  const totalQuestionSteps = Math.max(1, questionSteps.length);
  const currentQuestionNumber =
    currentStep === "submit"
      ? totalQuestionSteps
      : Math.max(1, questionSteps.indexOf(currentStep) + 1);

  const canGoNext = useMemo(() => {
    if (currentStep === "q1") return q1 !== "";
    if (currentStep === "q2") return q2 !== "";
    if (currentStep === "favorite") {
      if (textsLoading) return false;
      if (textsError) return true;
      if (favoriteOptions.length === 0) return true;
      return favoriteTextId !== "";
    }
    if (currentStep === "favoriteWhy") return favoriteWhy !== "";
    return false;
  }, [currentStep, favoriteOptions.length, favoriteTextId, favoriteWhy, q1, q2, textsError, textsLoading]);

  const handleNext = () => {
    if (currentStep === "submit") return;
    if (!canGoNext) return;
    setStepIndex((i) => Math.min(i + 1, steps.length - 1));
  };

  const handlePrev = () => {
    if (currentStep === "submit") {
      setStepIndex(Math.max(0, steps.length - 2));
      return;
    }
    setStepIndex((i) => Math.max(0, i - 1));
  };

  const handleFinish = async () => {
    if (!canSubmit) return;

    const favoriteTextIdPayload =
      favoriteTextId === FAVORITE_NONE ? null : favoriteTextId || null;
    const favoriteWhyPayload = favoriteTextIdPayload ? favoriteWhy || null : null;

    setSubmitting(true);
    try {
      await submitSessionFeedback({
        q1: Number.parseInt(q1, 10),
        q2,
        favorite_text_id: favoriteTextIdPayload,
        favorite_why: favoriteWhyPayload,
      });

      await endSession();
      navigate("/");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#faf8f5]">
      <TopBar />

      <div className="h-8" />

      <main className="flex-1 container mx-auto px-4 pb-14 max-w-3xl">
        <Card className="p-8 md:p-10 mb-8 bg-white border border-[#e0ddd5] shadow-sm">
          <h1 className="text-3xl font-bold text-[#2d3142] leading-tight">
            Tilbakemelding
          </h1>

          <div className="mt-4">
            {currentStep === "submit" ? (
              <div className="flex flex-col items-center justify-center py-10 space-y-6 text-center">
                <h2 className="text-xl font-semibold text-[#2d3142]">
                  Takk! Vil du avslutte økten nå?
                </h2>

                <div className="flex flex-col gap-4 w-full max-w-xs">
                  <Button
                    onClick={handleFinish}
                    disabled={!canSubmit || submitting}
                  >
                    Avslutt
                  </Button>

                  <button
                    onClick={handlePrev}
                    className="text-[#5d6875] hover:text-[#4ecdc4] transition-colors"
                    type="button"
                  >
                    Tilbake til spørsmålene
                  </button>
                </div>
              </div>
            ) : (
              <QuestionContainer
                step={currentQuestionNumber}
                totalSteps={totalQuestionSteps}
                onNext={handleNext}
                onPrev={handlePrev}
                showPrev={stepIndex > 0}
                nextDisabled={!canGoNext}
              >
                {currentStep === "q1" && (
                  <div className="space-y-6">
                    <div className="flex items-start gap-3">
                      <Badge number={1} />
                      <h2 className="text-xl font-semibold text-[#2d3142] mt-1 leading-snug">
                        Viste programmet deg tekster du hadde lyst til å lese?
                      </h2>
                    </div>

                    <div className="pl-10">
                      <RadioGroup
                        name="feedback-q1"
                        value={q1}
                        onChange={setQ1}
                        className="mb-8"
                        options={[
                          { value: "1", label: "Nesten ingen av tekstene" },
                          { value: "2", label: "Noen av tekstene" },
                          { value: "4", label: "De fleste tekstene" },
                          { value: "5", label: "Alle tekstene" },
                        ]}
                      />
                    </div>
                  </div>
                )}

                {currentStep === "q2" && (
                  <div className="space-y-6">
                    <div className="flex items-start gap-3">
                      <Badge number={2} />
                      <h2 className="text-xl font-semibold text-[#2d3142] mt-1 leading-snug">
                        Hvordan synes du tekstene passet til nivået ditt?
                      </h2>
                    </div>

                    <div className="pl-10">
                      <RadioGroup
                        name="feedback-q2"
                        value={q2}
                        onChange={setQ2}
                        className="mb-8"
                        options={[
                          { value: "too_easy", label: "De fleste tekstene var for lette" },
                          {
                            value: "fit_ok",
                            label: "De fleste tekstene var passe vanskelige",
                          },
                          { value: "too_hard", label: "De fleste tekstene var for vanskelige" },
                          {
                            value: "varied",
                            label: "Det var veldig forskjellig fra tekst til tekst",
                          },
                        ]}
                      />
                    </div>
                  </div>
                )}

                {currentStep === "favorite" && (
                  <div className="space-y-6">
                    <div className="flex items-start gap-3">
                      <Badge number={3} />
                      <div className="space-y-2">
                        <h2 className="text-xl font-semibold text-[#2d3142] mt-1 leading-snug">
                          Hvilken tekst likte du best i dag?
                        </h2>
                      </div>
                    </div>

                    <div className="pl-10">
                      {textsLoading ? (
                        <p className="text-sm text-[#5d6875]">Laster tekster...</p>
                      ) : textsError ? (
                        <p className="text-sm text-red-600">{textsError}</p>
                      ) : favoriteOptions.length === 0 ? (
                        <p className="text-sm text-[#5d6875]">
                          Ingen tekster registrert i denne økten.
                        </p>
                      ) : (
                        <div className="grid grid-cols-2 gap-4 mb-8">
                          {favoriteOptions.map((opt, idx) => {
                            const selected = favoriteTextId === opt.text_id;
                            const isLastOdd =
                              favoriteOptions.length % 2 === 1 &&
                              idx === favoriteOptions.length - 1;

                            const resolvedImageUrl = resolveImageUrl(opt.first_image_url);

                            return (
                              <label
                                key={opt.text_id}
                                className={
                                  "flex items-start gap-3 p-4 rounded-[16px] cursor-pointer border transition-all duration-200 " +
                                  (selected
                                    ? "border-[#4ecdc4] bg-white shadow-[0_2px_8px_rgba(78,205,196,0.15)]"
                                    : "border-[#e0ddd5] bg-transparent hover:bg-white/50") +
                                  (isLastOdd
                                    ? " col-span-2 justify-self-center w-full max-w-[520px]"
                                    : "")
                                }
                              >
                                <div
                                  className={
                                    "mt-1 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors flex-shrink-0 " +
                                    (selected ? "border-[#4ecdc4]" : "border-[#e0ddd5]")
                                  }
                                >
                                  {selected && (
                                    <div className="w-2.5 h-2.5 rounded-full bg-[#4ecdc4]" />
                                  )}
                                </div>

                                <input
                                  type="radio"
                                  name="feedback-qa"
                                  value={opt.text_id}
                                  checked={selected}
                                  onChange={() => {
                                    setFavoriteTextId(opt.text_id);
                                    setFavoriteWhy("");
                                  }}
                                  className="hidden"
                                />

                                {resolvedImageUrl ? (
                                  <div className="w-20 h-16 rounded-[12px] overflow-hidden border border-[#e0ddd5] flex-shrink-0">
                                    <img
                                      src={resolvedImageUrl}
                                      alt={opt.title}
                                      loading="lazy"
                                      className="w-full h-full object-cover"
                                    />
                                  </div>
                                ) : null}

                                <span className="text-[#2d3142] font-medium">
                                  {opt.title}
                                </span>
                              </label>
                            );
                          })}
                          <label
                            className={
                              "flex items-start gap-3 p-4 rounded-[16px] cursor-pointer border transition-all duration-200 col-span-2 justify-self-center w-full max-w-[520px] " +
                              (favoriteTextId === FAVORITE_NONE
                                ? "border-[#4ecdc4] bg-white shadow-[0_2px_8px_rgba(78,205,196,0.15)]"
                                : "border-[#e0ddd5] bg-transparent hover:bg-white/50")
                            }
                          >
                            <div
                              className={
                                "mt-1 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors flex-shrink-0 " +
                                (favoriteTextId === FAVORITE_NONE
                                  ? "border-[#4ecdc4]"
                                  : "border-[#e0ddd5]")
                              }
                            >
                              {favoriteTextId === FAVORITE_NONE && (
                                <div className="w-2.5 h-2.5 rounded-full bg-[#4ecdc4]" />
                              )}
                            </div>

                            <input
                              type="radio"
                              name="feedback-qa"
                              value={FAVORITE_NONE}
                              checked={favoriteTextId === FAVORITE_NONE}
                              onChange={() => {
                                setFavoriteTextId(FAVORITE_NONE);
                                setFavoriteWhy("");
                              }}
                              className="hidden"
                            />

                            <span className="text-[#2d3142] font-medium">
                              Ingen
                            </span>
                          </label>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {currentStep === "favoriteWhy" && (
                  <div className="space-y-6">
                    <div className="flex items-start gap-3">
                      <Badge number={4} />
                      <h2 className="text-xl font-semibold text-[#2d3142] mt-1 leading-snug">
                        Hvorfor likte du den teksten best?
                      </h2>
                    </div>

                    <div className="pl-10">
                      <RadioGroup
                        name="feedback-qb"
                        value={favoriteWhy}
                        onChange={setFavoriteWhy}
                        className="mb-8"
                        options={[
                          {
                            value: "interested_in_topic",
                            label: "Det handlet om noe jeg er interessert i",
                          },
                          {
                            value: "understood_and_read",
                            label: "Jeg forstod den godt og klarte å lese den",
                          },
                          {
                            value: "learned_something",
                            label: "Jeg lærte noe jeg ikke visste fra før",
                          },
                          {
                            value: "more_exciting",
                            label: "Den var mer spennende enn de andre tekstene",
                          },
                        ]}
                      />
                    </div>
                  </div>
                )}
              </QuestionContainer>
            )}
          </div>
        </Card>
      </main>
    </div>
  );
}

function resolveImageUrl(url?: string | null): string | null {
  if (!url) return null;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${BACKEND_URL}${url}`;
  return `${BACKEND_URL}/${url}`;
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
  nextDisabled = false,
}: {
  children: React.ReactNode;
  step: number;
  totalSteps: number;
  onNext: () => void;
  onPrev: () => void;
  showPrev?: boolean;
  nextDisabled?: boolean;
}) {
  return (
    <div className="flex flex-col min-h-[420px] gap-0">
      <div className="flex-1 pt-4">{children}</div>

      <div className="pt-6 border-t border-[#e0ddd5] grid grid-cols-3 items-center">
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
          <Button onClick={onNext} disabled={nextDisabled}>
            Neste
          </Button>
        </div>
      </div>
    </div>
  );
}
