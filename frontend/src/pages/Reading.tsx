import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router";
import { TopBar } from "../components/ui/TopBar";
import { Card } from "../components/ui/card";
import { Button } from "../components/ui/Button";
import { RadioGroup } from "../components/ui/RadioGroup";
import { TextArea } from "../components/ui/textarea";
import { ArrowLeft, ArrowRight, CheckCircle2 } from "lucide-react";
import {
  BACKEND_URL,
  getTextById,
  getTextQuestionsById,
  logSessionEvent,
  submitReadingAnswers,
  type TextQuestion,
} from "../lib/session";
import { motion, AnimatePresence } from "motion/react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

type PanelVariant = "collapsed" | "question" | "submit";

type TextDetail = {
  text_id: string;
  title: string;
  body?: string | null;
  content?: string | null;
  image_urls?: string[];
  first_image_url?: string | null;
};

function normalizeTitleLikeText(value: string): string {
  return value
    .replace(/^\uFEFF/, "")
    .trim()
    .replace(/^['"“”‘’]+|['"“”‘’]+$/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function stripLeadingTitleHeading(markdown: string, title: string): string {
  if (!markdown) return "";
  const normalizedTitle = normalizeTitleLikeText(title);
  if (!normalizedTitle) return markdown;

  const lines = markdown.replace(/^\uFEFF/, "").split(/\r?\n/);
  let firstNonEmptyIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== "") {
      firstNonEmptyIndex = i;
      break;
    }
  }
  if (firstNonEmptyIndex === -1) return markdown;

  const firstLine = lines[firstNonEmptyIndex];
  const match = firstLine.match(/^(#{1,6})\s+(.*)$/);
  if (!match) return markdown;

  const headingText = normalizeTitleLikeText(match[2] ?? "");
  if (headingText !== normalizedTitle) return markdown;

  lines.splice(firstNonEmptyIndex, 1);
  return lines.join("\n").replace(/^\s*\n+/, "");
}

function removeEmptyMarkdownHeadings(markdown: string): string {
  if (!markdown) return "";
  return markdown.replace(/^#{1,6}\s*$/gm, "");
}

function resolveImageUrl(url?: string | null): string | null {
  if (!url) return null;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${BACKEND_URL}${url}`;
  return `${BACKEND_URL}/${url}`;
}

function resolveMarkdownImageSrc(src?: string): string | undefined {
  if (!src) return undefined;
  if (
    src.startsWith("http://") ||
    src.startsWith("https://") ||
    src.startsWith("data:")
  ) {
    return src;
  }
  if (src.startsWith("/")) return `${BACKEND_URL}${src}`;

  const trimmed = src.replace(/^\.\//, "");
  const fileName = trimmed.includes("/") ? trimmed.split("/").pop() : trimmed;
  if (!fileName) return src;
  return `${BACKEND_URL}/images/${encodeURIComponent(fileName)}`;
}

function resolveThumbnailUrl(url?: string | null): string | null {
  if (!url || url.startsWith("data:")) return null;

  try {
    const resolved = new URL(url, BACKEND_URL);
    if (
      !resolved.pathname.startsWith("/images/") ||
      resolved.pathname.startsWith("/images/thumbs/")
    ) {
      return null;
    }

    resolved.pathname = resolved.pathname.replace(
      "/images/",
      "/images/thumbs/"
    );
    return resolved.toString();
  } catch {
    return null;
  }
}

function enhanceHtmlContentImages(html: string): string {
  if (!html || typeof window === "undefined") return html;

  const parser = new DOMParser();
  const doc = parser.parseFromString(html, "text/html");
  const images = Array.from(doc.querySelectorAll("img"));

  images.forEach((image, index) => {
    const rawSrc = image.getAttribute("src");
    if (!rawSrc) return;

    const resolvedSrc = resolveMarkdownImageSrc(rawSrc);
    if (!resolvedSrc) return;

    const thumbnailSrc = resolveThumbnailUrl(resolvedSrc);
    image.setAttribute("src", thumbnailSrc ?? resolvedSrc);
    image.setAttribute("data-original-src", resolvedSrc);
    image.setAttribute("loading", index === 0 ? "eager" : "lazy");
    image.setAttribute("fetchpriority", index === 0 ? "high" : "auto");
    image.setAttribute("decoding", "async");
  });

  return doc.body.innerHTML;
}

function ReadingImage({
  src,
  alt,
  prioritize = false,
  ...props
}: React.ImgHTMLAttributes<HTMLImageElement> & {
  prioritize?: boolean;
}) {
  const fallbackSrc = src ?? null;
  const thumbnailSrc = resolveThumbnailUrl(fallbackSrc);
  const preferredSrc = thumbnailSrc ?? fallbackSrc;
  const [currentSrc, setCurrentSrc] = useState(preferredSrc ?? "");

  useEffect(() => {
    setCurrentSrc(preferredSrc ?? "");
  }, [preferredSrc]);

  if (!fallbackSrc) return null;

  return (
    <img
      {...props}
      src={currentSrc}
      alt={alt}
      loading={prioritize ? "eager" : "lazy"}
      fetchPriority={prioritize ? "high" : "auto"}
      decoding="async"
      onError={() => {
        if (currentSrc !== fallbackSrc) {
          setCurrentSrc(fallbackSrc);
        }
      }}
    />
  );
}

function fillEmptyMarkdownImageUrls(markdown: string, imageUrls: string[]): string {
  if (!markdown) return "";
  if (!Array.isArray(imageUrls) || imageUrls.length === 0) return markdown;

  let index = 0;
  return markdown.replace(/!\[([^\]]*)\]\(\s*\)/g, (_match, alt: string) => {
    const next = imageUrls[index++];
    if (!next) return `![${alt}]()`;
    return `![${alt}](${next})`;
  });
}

export function Reading() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [variant, setVariant] = useState<PanelVariant>("collapsed");
  const [isDesktop, setIsDesktop] = useState(false);

  const [questions, setQuestions] = useState<TextQuestion[]>([]);
  const [questionsLoading, setQuestionsLoading] = useState(false);
  const [questionsError, setQuestionsError] = useState<string | null>(null);
  const [questionIndex, setQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string | string[]>>({});
  const [submitting, setSubmitting] = useState(false);

  const [book, setBook] = useState<TextDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const leftScrollRef = useRef<HTMLDivElement | null>(null);
  const rightCardRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    setError(null);
    getTextById(id)
      .then((data) => {
        setBook(data);
        setLoading(false);
      })
      .catch(() => {
        setError("Kunne ikke hente teksten.");
        setLoading(false);
      });
  }, [id]);

  useEffect(() => {
    const media = window.matchMedia("(min-width: 1024px)");
    const update = () => setIsDesktop(media.matches);
    update();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }

    media.addListener(update);
    return () => media.removeListener(update);
  }, []);

  useEffect(() => {
    if (!isDesktop) return;

    const handleWheel = (e: WheelEvent) => {
      const leftEl = leftScrollRef.current;
      const rightEl = rightCardRef.current;

      if (!leftEl) return;

      if (rightEl) {
        const rect = rightEl.getBoundingClientRect();
        const isInsideRight =
          e.clientX >= rect.left &&
          e.clientX <= rect.right &&
          e.clientY >= rect.top &&
          e.clientY <= rect.bottom;

        if (isInsideRight) return;
      }

      const canScroll = leftEl.scrollHeight > leftEl.clientHeight;
      if (!canScroll) return;

      const atTop = leftEl.scrollTop <= 0;
      const atBottom =
        Math.ceil(leftEl.scrollTop + leftEl.clientHeight) >= leftEl.scrollHeight;

      if ((e.deltaY < 0 && atTop) || (e.deltaY > 0 && atBottom)) {
        return;
      }

      e.preventDefault();
      leftEl.scrollTop += e.deltaY;
    };

    window.addEventListener("wheel", handleWheel, { passive: false });

    return () => {
      window.removeEventListener("wheel", handleWheel);
    };
  }, [isDesktop]);

  const handleShowQuestions = async () => {
    if (!id) return;

    void logSessionEvent({
      event_type: "reading_questions_opened",
      text_id: id,
    }).catch(() => {
      // best-effort analytics
    });

    setQuestionsError(null);
    setQuestionsLoading(true);
    try {
      const fetched = await getTextQuestionsById(id);
      setQuestions(fetched);
      setQuestionIndex(0);
      setVariant(fetched.length > 0 ? "question" : "submit");
    } catch {
      setQuestionsError("Kunne ikke hente spørsmålene.");
    } finally {
      setQuestionsLoading(false);
    }
  };

  const handleNext = () => {
    if (variant === "collapsed") {
      void handleShowQuestions();
      return;
    }

    if (variant === "question") {
      if (questionIndex < questions.length - 1) setQuestionIndex((i) => i + 1);
      else setVariant("submit");
    }
  };

  const handlePrev = () => {
    if (variant === "question") {
      if (questionIndex > 0) setQuestionIndex((i) => i - 1);
      else setVariant("collapsed");
      return;
    }

    if (variant === "submit") {
      if (questions.length > 0) {
        setVariant("question");
        setQuestionIndex(Math.max(0, questions.length - 1));
      } else {
        setVariant("collapsed");
      }
    }
  };

  const handleSubmit = async () => {
    if (!id) return;

    setSubmitting(true);
    try {
      await submitReadingAnswers({
        text_id: id,
        answers,
      });
    } catch {
      // Keep UX simple: still allow the student to proceed.
      // If you want strict persistence, we can block navigation and show an error.
    } finally {
      setSubmitting(false);
      navigate("/completion");
    }
  };

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center">Laster...</div>;
  }

  if (error || !book) {
    return (
      <div className="min-h-screen flex items-center justify-center text-red-600">
        {error || "Tekst ikke funnet."}
      </div>
    );
  }

  const resolvedFirstImageUrl = resolveImageUrl(book.first_image_url);

  const content: string =
    typeof book.body === "string"
      ? book.body
      : typeof book.content === "string"
        ? book.content
        : "";

  const contentLooksLikeHtml = /<\/?[a-z][\s\S]*>/i.test(content);
  const cleanedContent = removeEmptyMarkdownHeadings(
    stripLeadingTitleHeading(content, book.title)
  );
  const contentHasMarkdownImages = /!\[[^\]]*\]\(\s*[^)]*\)/.test(cleanedContent);
  const contentHasHtmlImages = /<img\b/i.test(content);
  const contentHasAnyImages = contentLooksLikeHtml
    ? contentHasHtmlImages
    : contentHasMarkdownImages;

  const markdown = fillEmptyMarkdownImageUrls(
    cleanedContent,
    Array.isArray(book.image_urls) ? book.image_urls : []
  );
  const htmlContent = contentLooksLikeHtml
    ? enhanceHtmlContentImages(content)
    : content;

  let prioritizedMarkdownImageRendered = false;
  const markdownComponents: Components = {
    h2: ({ children, ...props }) => (
      <h2
        {...props}
        className="text-2xl font-semibold"
        style={{
          fontSize: "1.75rem",
          lineHeight: 1.25,
          marginTop: "1.5rem",
          marginBottom: "0.5rem",
        }}
      >
        {children}
      </h2>
    ),
    h3: ({ children, ...props }) => (
      <h3
        {...props}
        className="text-xl font-semibold"
        style={{
          fontSize: "1.5rem",
          lineHeight: 1.3,
          marginTop: "1.25rem",
          marginBottom: "0.5rem",
        }}
      >
        {children}
      </h3>
    ),
    p: ({ children, ...props }) => {
      const parts = React.Children.toArray(children).filter((child) => {
        if (typeof child === "string") return child.trim() !== "";
        return true;
      });

      const isOnlyImage =
        parts.length === 1 &&
        React.isValidElement(parts[0]) &&
        parts[0].type === "img";

      if (isOnlyImage) {
        return (
          <p {...props} style={{ margin: 0 }}>
            {children}
          </p>
        );
      }

      return (
        <p {...props} style={{ marginTop: "0.75rem", marginBottom: "1rem" }}>
          {children}
        </p>
      );
    },
    img: ({ src, alt, ...props }) => {
      const resolved = resolveMarkdownImageSrc(src);
      if (!resolved) return null;

      const shouldPrioritize = !prioritizedMarkdownImageRendered;
      prioritizedMarkdownImageRendered = true;

      return (
        <ReadingImage
          {...props}
          src={resolved}
          alt={alt || ""}
          prioritize={shouldPrioritize}
          style={{
            display: "block",
            maxWidth: "100%",
            height: "auto",
            marginTop: "1.5rem",
            marginBottom: "1.5rem",
          }}
        />
      );
    },
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar />

      <main className="flex-1 container mx-auto px-4 py-8 max-w-[1440px]">
        <div className="grid lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7">
            <div
              ref={leftScrollRef}
              className="no-scrollbar"
              style={
                isDesktop
                  ? {
                      height: "calc(100vh - 128px)",
                      overflowY: "auto",
                      overflowX: "hidden",
                    }
                  : undefined
              }
            >
              <Card className="p-8 border border-[#e0ddd5]">
                <h1
                  className="text-3xl font-bold text-[#2d3142] mb-6"
                  style={{ fontSize: "2.5rem", lineHeight: 1.12 }}
                >
                  {book.title}
                </h1>

                {resolvedFirstImageUrl && !contentHasAnyImages && (
                  <div className="w-full h-64 md:h-80 rounded-[24px] overflow-hidden shadow-md mb-6">
                    <ReadingImage
                      src={resolvedFirstImageUrl}
                      alt={book.title}
                      className="w-full h-full object-cover"
                      prioritize
                    />
                  </div>
                )}

                {contentLooksLikeHtml ? (
                  <div
                    className="prose prose-lg prose-slate max-w-none text-[#2d3142] leading-relaxed"
                    style={{ fontSize: "1.25rem", lineHeight: 1.7 }}
                    dangerouslySetInnerHTML={{ __html: htmlContent }}
                  />
                ) : (
                  <div
                    className="prose prose-lg prose-slate max-w-none text-[#2d3142] leading-relaxed"
                    style={{ fontSize: "1.25rem", lineHeight: 1.7 }}
                  >
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={markdownComponents}
                    >
                      {markdown}
                    </ReactMarkdown>
                  </div>
                )}
              </Card>
            </div>
          </div>

          <div className="lg:col-span-5">
            <div
              style={
                isDesktop
                  ? {
                      position: "sticky",
                      top: "96px",
                    }
                  : undefined
              }
            >
              <div ref={rightCardRef}>
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
                        <Button onClick={handleShowQuestions} disabled={questionsLoading}>
                          Vis spørsmål <ArrowRight size={18} />
                        </Button>

                        {questionsLoading && (
                          <p className="text-sm text-[#5d6875]">Laster spørsmål...</p>
                        )}

                        {questionsError && (
                          <p className="text-sm text-red-600">{questionsError}</p>
                        )}
                      </motion.div>
                    )}

                    {variant === "question" && (
                      <QuestionContainer
                        key={`question-${questions[questionIndex]?.question_id ?? questionIndex}`}
                        step={questions.length > 0 ? questionIndex + 1 : 1}
                        totalSteps={Math.max(1, questions.length)}
                        onNext={handleNext}
                        onPrev={handlePrev}
                      >
                        {questions.length === 0 ? (
                          <div className="space-y-2 text-center">
                            <h3 className="text-xl font-semibold text-[#2d3142]">
                              Ingen spørsmål for denne teksten.
                            </h3>
                            <p className="text-sm text-[#5d6875]">
                              Trykk Neste for å fortsette.
                            </p>
                          </div>
                        ) : (
                          (() => {
                            const q = questions[questionIndex];
                            const rawAnswer = answers[q.question_id];
                            const textValue = typeof rawAnswer === "string" ? rawAnswer : "";
                            const singleValue = typeof rawAnswer === "string" ? rawAnswer : "";
                            const multiValue = Array.isArray(rawAnswer) ? rawAnswer : [];
                            const hasOptions = Array.isArray(q.options) && q.options.length > 0;
                            const normalizedType = (q.question_type ?? "").trim().toLowerCase();
                            const isTrueOrFalse = normalizedType === "trueorfalse";
                            const isCheckbox =
                              normalizedType === "checkboxes" || normalizedType === "checkbox";

                            return (
                              <div className="space-y-6">
                                <div className="flex items-start gap-3">
                                  <Badge number={questionIndex + 1} />
                                  <h3 className="text-xl font-semibold text-[#2d3142] mt-1">
                                    {q.body}
                                  </h3>
                                </div>

                                <div className="pl-12 space-y-2">
                                  {hasOptions && isTrueOrFalse ? (
                                    <div className="space-y-4">
                                      {q.options.map((opt) => {
                                        const key = `${q.question_id}:${opt.option_id}`;
                                        const tfRaw = answers[key];
                                        const tfValue = typeof tfRaw === "string" ? tfRaw : "";

                                        return (
                                          <div
                                            key={opt.option_id}
                                            className="bg-white p-4 rounded-[16px] border border-[#e0ddd5]"
                                          >
                                            <p className="mb-3 text-[#2d3142] font-medium">
                                              {opt.body}
                                            </p>
                                            <div className="flex gap-6">
                                              <label className="flex items-center gap-2 cursor-pointer">
                                                <input
                                                  type="radio"
                                                  name={`tf-${q.question_id}-${opt.option_id}`}
                                                  className="w-4 h-4 accent-[#4ecdc4]"
                                                  checked={tfValue === "true"}
                                                  onChange={() =>
                                                    setAnswers((prev) => ({
                                                      ...prev,
                                                      [key]: "true",
                                                    }))
                                                  }
                                                />
                                                <span>Rett</span>
                                              </label>
                                              <label className="flex items-center gap-2 cursor-pointer">
                                                <input
                                                  type="radio"
                                                  name={`tf-${q.question_id}-${opt.option_id}`}
                                                  className="w-4 h-4 accent-[#4ecdc4]"
                                                  checked={tfValue === "false"}
                                                  onChange={() =>
                                                    setAnswers((prev) => ({
                                                      ...prev,
                                                      [key]: "false",
                                                    }))
                                                  }
                                                />
                                                <span>Galt</span>
                                              </label>
                                            </div>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  ) : hasOptions && isCheckbox ? (
                                    <div className="space-y-3">
                                      {q.options.map((opt) => {
                                        const checked = multiValue.includes(opt.option_id);
                                        return (
                                          <div
                                            key={opt.option_id}
                                            className="bg-white p-4 rounded-[16px] border border-[#e0ddd5]"
                                          >
                                            <label className="flex items-start gap-3 cursor-pointer">
                                              <input
                                                type="checkbox"
                                                className="mt-1 w-4 h-4 accent-[#4ecdc4]"
                                                checked={checked}
                                                onChange={() =>
                                                  setAnswers((prev) => {
                                                    const prevRaw = prev[q.question_id];
                                                    const selected = Array.isArray(prevRaw)
                                                      ? prevRaw
                                                      : [];
                                                    const isChecked = selected.includes(opt.option_id);
                                                    const next = isChecked
                                                      ? selected.filter((id) => id !== opt.option_id)
                                                      : [...selected, opt.option_id];
                                                    return { ...prev, [q.question_id]: next };
                                                  })
                                                }
                                              />
                                              <span className="text-[#2d3142] font-medium">
                                                {opt.body}
                                              </span>
                                            </label>
                                          </div>
                                        );
                                      })}
                                    </div>
                                  ) : hasOptions ? (
                                    <RadioGroup
                                      name={`q-${q.question_id}`}
                                      value={singleValue}
                                      onChange={(next) =>
                                        setAnswers((prev) => ({
                                          ...prev,
                                          [q.question_id]: next,
                                        }))
                                      }
                                      options={q.options.map((opt) => ({
                                        value: opt.option_id,
                                        label: opt.body,
                                      }))}
                                    />
                                  ) : (
                                    <>
                                      <TextArea
                                        placeholder="Skriv svaret ditt her..."
                                        maxLength={190}
                                        value={textValue}
                                        onChange={(e) =>
                                          setAnswers((prev) => ({
                                            ...prev,
                                            [q.question_id]: e.target.value,
                                          }))
                                        }
                                        className="h-32"
                                      />
                                      <div className="text-right text-xs text-[#5d6875]">
                                        {textValue.length} / 190 tegn
                                      </div>
                                    </>
                                  )}
                                </div>
                              </div>
                            );
                          })()
                        )}
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
                          <CheckCircle2 size={32} className="text-[#e07a5f]" />
                        </div>
                        <h3 className="text-xl font-semibold text-[#2d3142]">
                          Nå har du gått gjennom alle spørsmålene, vil du sende inn?
                        </h3>

                        <div className="flex flex-col gap-4 w-full max-w-xs">
                          <Button variant="completion" onClick={handleSubmit} disabled={submitting}>
                            Send inn
                          </Button>
                          <button
                            onClick={handlePrev}
                            className="text-[#5d6875] hover:text-[#4ecdc4] flex items-center justify-center gap-2 transition-colors"
                          >
                            <ArrowLeft size={16} /> Tilbake til spørsmålene
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Card>
              </div>
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