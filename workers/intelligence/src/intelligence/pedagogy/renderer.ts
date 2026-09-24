import type { ReasoningResult } from "../../schemas";

export function renderReasoning(result: ReasoningResult, depth: "BRIEF" | "NORMAL" | "DEEP" = "NORMAL"): string {
  if (result.status === "insufficient_evidence") {
    const reason = result.uncertaintyReason ?? "The available evidence does not support a reliable answer.";
    return result.nextAction ? `${reason} ${result.nextAction}` : reason;
  }
  const claims = result.claims.filter((claim) => claim.verificationStatus === "verified").map((claim) => claim.text);
  if (claims.length === 0) return "I don't have enough verified information to answer that reliably.";
  if (depth === "BRIEF") return claims[0] ?? "";
  const main = claims.join(depth === "DEEP" ? "\n\n" : " ");
  const misconception = result.misconception ? `\n\nThe key misconception is: ${result.misconception.text}` : "";
  const action = result.nextAction ? `\n\nNext: ${result.nextAction}` : "";
  return `${main}${misconception}${action}`;
}
