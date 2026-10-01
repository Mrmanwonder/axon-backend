interface DocumentVisionSecretBindings {
  GOOGLE_API_KEY: string;
  /** Deploy-time configuration (a var set at deploy), never a literal in code or wrangler.jsonc. */
  GEMINI_MODEL: string;
}

type Env = DocumentVisionBindings & DocumentVisionSecretBindings;

declare namespace Cloudflare {
  // Declaration merging makes secret bindings visible to the module Worker.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface Env extends DocumentVisionSecretBindings {}
}
