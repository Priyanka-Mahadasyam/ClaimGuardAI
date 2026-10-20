# Multiple-claim upload support

The Check a claim page now supports assessing multiple claims from one upload:

- **PDF:** each text-readable page is treated as a separate claim. Multiple repeated claim sections on one page are also split when they use a `Claim Number`, `Claim ID`, or `Claim No` label.
- **CSV / Excel:** each non-empty data row is treated as a separate claim.
- **JSON:** a list of claim objects is supported; each object is assessed separately.

The app shows a batch summary and then renders an individual assessment for each claim. If one claim is missing required fields, that claim is marked as not assessed and the remaining valid claims are still processed.

Scanned/image-only PDFs still require OCR before the claim fields can be read. PDF pages should contain one claim each for the most reliable results.
