Examine the sampled frames from a Manim-rendered educational video and evaluate each criterion below. Respond with a structured object containing a boolean for each field — `true` if the problem is present, `false` if it is not.

Criteria:
- **text_clipped**: Are any text labels or equations clipped or cut off at the frame edges?
- **overlapping_content**: Is any content overlapping or rendered unreadably on top of other content?
- **broken_animations**: Are there any broken or glitchy visual artifacts, or misplaced objects?
- **content_overflow**: Is any content overflowing / extending outside the visible frame boundary?
- **latex_rendering**: Is any LaTeX incorrectly rendered (broken symbols, blank boxes, malformed equations)?

Be strict: flag `true` for any criterion where you see even a single clear example.
