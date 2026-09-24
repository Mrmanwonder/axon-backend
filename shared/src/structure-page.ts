import type { SupabaseClient } from '@supabase/supabase-js';
import type { RawMark } from './attribution.js';
import { mustMaybe } from './db.js';

// Only persisted columns belong here. margin_band is device-local and is not
// part of paper_page; selecting it makes PostgREST reject the entire query.
export const STRUCTURE_PAGE_COLUMNS = 'id, paper_id, student_id, page_number, r2_bucket, r2_key, mask_key, structure_status, layer_fallback, teacher_marks, conditioning_meta, quality_signals';

export interface StructurePage {
  id: string;
  paper_id: string;
  student_id: string;
  page_number: number;
  r2_bucket: string | null;
  r2_key: string;
  mask_key: string | null;
  structure_status: string;
  layer_fallback: string | null;
  teacher_marks: RawMark[] | null;
  conditioning_meta: Record<string, unknown> | null;
  quality_signals: Record<string, unknown> | null;
}

export function loadStructurePage(sb: SupabaseClient, pageId: string) {
  return mustMaybe<StructurePage>(
    sb.from('paper_page').select(STRUCTURE_PAGE_COLUMNS).eq('id', pageId).maybeSingle(),
    'structure page read',
  );
}
