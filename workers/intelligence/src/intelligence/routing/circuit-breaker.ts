export type CircuitState = "CLOSED" | "HALF_OPEN" | "OPEN";

export interface ProviderObservation {
  success: boolean;
  timeout?: boolean;
  rateLimited?: boolean;
  serverError?: boolean;
  schemaFailure?: boolean;
  semanticFailure?: boolean;
  latencyMs: number;
}

export class CircuitBreaker {
  #state: CircuitState = "CLOSED";
  #failures = 0;
  #openedAt = 0;
  constructor(readonly threshold = 5, readonly resetAfterMs = 30_000) {}

  state(now = Date.now()): CircuitState {
    if (this.#state === "OPEN" && now - this.#openedAt >= this.resetAfterMs) this.#state = "HALF_OPEN";
    return this.#state;
  }

  allow(now = Date.now()): boolean { return this.state(now) !== "OPEN"; }

  record(observation: ProviderObservation, now = Date.now()): void {
    if (observation.success) {
      this.#failures = 0;
      this.#state = "CLOSED";
      return;
    }
    this.#failures += 1;
    if (this.#state === "HALF_OPEN" || this.#failures >= this.threshold) {
      this.#state = "OPEN";
      this.#openedAt = now;
    }
  }
}
