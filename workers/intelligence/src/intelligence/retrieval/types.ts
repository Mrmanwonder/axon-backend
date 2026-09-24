import type { Evidence } from "../../schemas";

export interface RetrievalRequest {
  query: string;
  purpose: "current_fact" | "official_rule" | "academic_research" | "source_lookup";
  preferredDomains?: string[];
  requiredFreshness?: string;
  maxSources?: number;
}

export interface RetrievalService {
  retrieve(request: RetrievalRequest): Promise<Evidence[]>;
}
