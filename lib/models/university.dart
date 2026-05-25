// lib/models/university.dart

enum UniversityCategory { dream, reach, safety }

class University {
  final String id;
  final String name;
  final String location;
  final String description;
  final String logoUrl;
  final List<Degree> degrees;

  University({
    required this.id,
    required this.name,
    required this.location,
    required this.description,
    required this.logoUrl,
    required this.degrees,
  });
}

class Degree {
  final String id;
  final String name;
  final String duration;
  final String syllabus;
  final Map<String, String> gradeRequirements; // e.g., {"IB": "38", "A-Level": "A*AA"}
  final List<String> requiredSubjects;

  Degree({
    required this.id,
    required this.name,
    required this.duration,
    required this.syllabus,
    required this.gradeRequirements,
    required this.requiredSubjects,
  });
}

class UserUniversityPreference {
  final String universityId;
  final String degreeId;
  final UniversityCategory category;
  final DateTime addedAt;

  UserUniversityPreference({
    required this.universityId,
    required this.degreeId,
    required this.category,
    required this.addedAt,
  });
}

class UserAchievement {
  final String id;
  final String title;
  final String description;
  final DateTime date;
  final double importanceRating; // 0.0 to 1.0
  final String feedback;

  UserAchievement({
    required this.id,
    required this.title,
    required this.description,
    required this.date,
    required this.importanceRating,
    required this.feedback,
  });
}
