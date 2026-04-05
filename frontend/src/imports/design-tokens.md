1) Apply Design Tokens (must match)

Colors

Background primary: #faf8f5

Background secondary: #f5f1ea

Text primary: #2d3142

Text secondary: #5d6875

Border default: #e0ddd5

Hover background: #e0ddd5

Accent teal: #4ecdc4

Accent sage: #95b8a2

Accent coral: #ff6b6b

Accent terracotta: #e07a5f

Accent mustard: #f4a261

Accent deep purple: #6b5b95

Gradients

Primary action gradient (main buttons): linear-gradient(to right, #4ecdc4, #95b8a2)

Completion/retake gradient: linear-gradient(to right, #f4a261, #e07a5f)

Hover overlay gradient: linear-gradient(to right, #ff6b6b, #4ecdc4)

Decorative blobs at ~20% opacity:

cool: rgba(78,205,196,0.2) → rgba(149,184,162,0.2)

warm: rgba(255,107,107,0.2) → rgba(224,122,95,0.2)

Radii

Inputs/buttons: 16px

Cards/containers: 24px

Pills/badges: 9999px

Shadows

Small (buttons): 0 2px 8px rgba(0,0,0,0.1)

Medium (cards): 0 8px 30px rgba(0,0,0,0.12)

Large (hover): 0 12px 40px rgba(0,0,0,0.18)

Dashboard 3D card: layered shadow + 1px outline feel

Typography

Font family: Inter

H1 32px/700, H2 24px/600, H3 20px/600, Body 16px/regular line-height 1.6

Effects

Glass header option: white at 0.8 opacity + blur (12px)

Paper texture optional for big panels (subtle grid texture)

Background blobs with heavy blur (like blur-3xl)

2) Frames / Screens to generate (1440px wide)
Screen A — Login

Layout:

Centered hero: title “Read in ai•les!” + short description of app.

Under hero: login card on background #faf8f5.
Login card:

Role radio: Pupil (default) / Teacher

Username input, Password input

Primary gradient button: Log in

“Forgot password?” + small helper text.
Styling:

Card uses white surface, radius 24, medium shadow.

Inputs radius 16, border 1px #e0ddd5, focus border teal.
Interaction:

Log in → navigates to Screen B.

Screen B — Landing / Recommended texts

Top bar:

Left: “ai•les”

Right: user chip (avatar + name) and Log out button (mustard or secondary).
Header:

“Hi [Name]!” (H1)

Subtitle: “Choose the text you most want to read.”

Link on right: “I don’t want to read now.”
Content:

Two large recommendation cards side-by-side (radius 24, 3D shadow).
Each card:

Image top, title, 2–3 lines excerpt, primary gradient button I want to read this!
Hover:

Card lifts slightly (translateY -4) + large shadow.
Interaction:

Clicking either card → Screen C (reading page).

Screen C — Reading page (questions hidden by default)

Two-column layout:
Left column:

H1 title, image with small credit caption, readable article text.
Right column:

A big beige panel (#f5f1ea) radius 24, medium shadow.
Default state:

Primary gradient button: Show questions →
Interaction:

Show questions toggles the right panel into Question Panel state (no page navigation).

3) Question Panel (single component with variants)

Create one reusable component QuestionPanel with these variants (swap variants on Next/Previous, do not navigate pages):

Variant 0: Collapsed

Button: “Show questions →”

Variant 1: Q1 Multiple choice

Badge “1” (pill/circle)

Question text

Radio options

Footer: progress “1 of 4” + Next button (primary gradient)

Variant 2: Q2 Short answer

Badge “2”

Large text area + “Max 190 characters”

Footer: Previous (mustard or secondary) + Next (primary)

Progress “2 of 4”

Variant 3: Q3 True/False

Badge “3”

3 statements each with radio True/False

Footer Previous/Next

Progress “3 of 4”

Variant 4: Q4 Rating

Badge “4”

Question: “How much did you like the text?”

5 option rating circles decreasing in size with labels:

Very good, Well, Medium, Little, Very little

Footer Previous/Next

Progress “4 of 4”

Variant 5: Submit confirmation

Text: “Now that you’ve gone through all the questions, do you want to submit?”

Button: Submit (completion gradient mustard→terracotta)

Link: “← Back to the questions”

Interactions:

Next/Previous swaps variants.

Q4 Next → Submit confirmation.

Submit → Screen D.

Screen D — Completion / Bravo

Center content:

Big playful “BRAVO” heading (use accent colors)

Text: “You are finished with this text!”

Button: Find a new text (primary gradient)

Link: “I don’t want to read any more now.”
Decor:

Add blurred blob background shapes (cool + warm) at 20% opacity.
Interaction:

Find a new text → Screen B.

4) Components to build (and reuse)

Button (Primary gradient / Secondary / Link)

Card (Recommendation card)

Input (default/focus/error)

Radio group

Text area

Badge step indicator

Top bar (logo + user chip + log out)

QuestionPanel component variants

Optional progress bar using teal→sage gradient

Make sure all spacing uses an 8px scale and the UI matches the QuizQ color/gradient/radius/shadow system exactly.