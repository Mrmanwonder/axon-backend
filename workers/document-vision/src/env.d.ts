interface DocumentVisionSecretBindings {
  GOOGLE_API_KEY: string;
}

type Env = DocumentVisionBindings & DocumentVisionSecretBindings;

declare namespace Cloudflare {
  // Declaration merging makes secret bindings visible to the module Worker.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface Env extends DocumentVisionSecretBindings {}
}
