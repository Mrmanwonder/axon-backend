import 'package:flutter/material.dart';

import '../models/daily_plan_task.dart';

IconData taskTypeIcon(TaskType type) {
  switch (type) {
    case TaskType.deepWork: return Icons.psychology_rounded;
    case TaskType.practice: return Icons.fitness_center_rounded;
    case TaskType.review: return Icons.book_rounded;
    case TaskType.flashcards: return Icons.style_rounded;
    case TaskType.pastPaper: return Icons.assignment_rounded;
    case TaskType.mockExam: return Icons.emoji_events_rounded;
    case TaskType.commandWordDrill: return Icons.text_fields_rounded;
  }
}

Color taskTypeColor(TaskType type) {
  switch (type) {
    case TaskType.deepWork: return const Color(0xFF8B5CF6);
    case TaskType.practice: return const Color(0xFF3B82F6);
    case TaskType.review: return const Color(0xFF10B981);
    case TaskType.flashcards: return const Color(0xFFF59E0B);
    case TaskType.pastPaper: return const Color(0xFFE11D48);
    case TaskType.mockExam: return const Color(0xFFEC4899);
    case TaskType.commandWordDrill: return const Color(0xFF06B6D4);
  }
}

Color intensityColor(IntensityLevel level) {
  switch (level) {
    case IntensityLevel.red: return const Color(0xFFE11D48);
    case IntensityLevel.orange: return const Color(0xFFF59E0B);
    case IntensityLevel.blue: return const Color(0xFF3B82F6);
  }
}
