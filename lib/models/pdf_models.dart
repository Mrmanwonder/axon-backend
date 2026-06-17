class PdfQuestion {
  final int questionNumber;
  final String questionText;
  final List<PdfPart> parts;
  final double yPosition;
  final double xPosition;
  final double width;
  final double height;
  final int pageNumber;
  String? userAnswer;
  String? correctAnswer;
  String? feedback;
  bool? isCorrect;
  int? marksAvailable;
  int? marksAwarded;
  final String? contextText;
  final List<String> figurePaths;
  final List<PdfFigureRegion> figureRegions;
  final List<String>? figureBase64;
  final String subjectTag;
  final String chapterTag;
  final String topicTag;
  final String difficultyTag;
  final String paperType;
  final String boardTag;
  final String paperYear;
  final Map<String, dynamic> spatialMetadata;

  PdfQuestion({
    required this.questionNumber,
    required this.questionText,
    this.parts = const [],
    required this.yPosition,
    required this.xPosition,
    required this.width,
    required this.height,
    required this.pageNumber,
    this.userAnswer,
    this.correctAnswer,
    this.feedback,
    this.isCorrect,
    this.marksAvailable,
    this.marksAwarded,
    this.contextText,
    List<String> figurePaths = const [],
    List<PdfFigureRegion> figureRegions = const [],
    List<String>? figureBase64,
    this.subjectTag = '',
    this.chapterTag = '',
    this.topicTag = '',
    this.difficultyTag = '',
    this.paperType = '',
    this.boardTag = '',
    this.paperYear = '',
    Map<String, dynamic> spatialMetadata = const {},
  })  : figurePaths = List<String>.from(figurePaths),
        figureRegions = List<PdfFigureRegion>.from(figureRegions),
        figureBase64 =
            figureBase64 == null ? null : List<String>.from(figureBase64),
        spatialMetadata = Map<String, dynamic>.from(spatialMetadata);

  String get fullText {
    final partsText = parts.map((part) => part.text).join(' ');
    return [
      contextText ?? '',
      questionText,
      partsText,
    ].where((value) => value.trim().isNotEmpty).join(' ').trim();
  }

  String get toFullLaTeX {
    final segments = <String>[
      if ((contextText ?? '').trim().isNotEmpty) contextText!.trim(),
      if (questionText.trim().isNotEmpty) questionText.trim(),
      ...parts
          .where((part) => part.text.trim().isNotEmpty)
          .map((part) => r'\textbf{(' + part.label + r')} ' + part.text.trim()),
    ];
    return segments.join(r' \\ ').trim();
  }

  PdfQuestion copyWith({
    int? questionNumber,
    String? questionText,
    List<PdfPart>? parts,
    double? yPosition,
    double? xPosition,
    double? width,
    double? height,
    int? pageNumber,
    String? userAnswer,
    String? correctAnswer,
    String? feedback,
    bool? isCorrect,
    int? marksAvailable,
    int? marksAwarded,
    String? contextText,
    List<String>? figurePaths,
    List<PdfFigureRegion>? figureRegions,
    List<String>? figureBase64,
    String? subjectTag,
    String? chapterTag,
    String? topicTag,
    String? difficultyTag,
    String? paperType,
    String? boardTag,
    String? paperYear,
    Map<String, dynamic>? spatialMetadata,
  }) {
    return PdfQuestion(
      questionNumber: questionNumber ?? this.questionNumber,
      questionText: questionText ?? this.questionText,
      parts: parts ?? this.parts,
      yPosition: yPosition ?? this.yPosition,
      xPosition: xPosition ?? this.xPosition,
      width: width ?? this.width,
      height: height ?? this.height,
      pageNumber: pageNumber ?? this.pageNumber,
      userAnswer: userAnswer ?? this.userAnswer,
      correctAnswer: correctAnswer ?? this.correctAnswer,
      feedback: feedback ?? this.feedback,
      isCorrect: isCorrect ?? this.isCorrect,
      marksAvailable: marksAvailable ?? this.marksAvailable,
      marksAwarded: marksAwarded ?? this.marksAwarded,
      contextText: contextText ?? this.contextText,
      figurePaths: figurePaths ?? this.figurePaths,
      figureRegions: figureRegions ?? this.figureRegions,
      figureBase64: figureBase64 ?? this.figureBase64,
      subjectTag: subjectTag ?? this.subjectTag,
      chapterTag: chapterTag ?? this.chapterTag,
      topicTag: topicTag ?? this.topicTag,
      difficultyTag: difficultyTag ?? this.difficultyTag,
      paperType: paperType ?? this.paperType,
      boardTag: boardTag ?? this.boardTag,
      paperYear: paperYear ?? this.paperYear,
      spatialMetadata: spatialMetadata ?? this.spatialMetadata,
    );
  }

  factory PdfQuestion.fromJson(Map<String, dynamic> json) {
    int asInt(dynamic v) {
      if (v is int) return v;
      if (v is num) return v.toInt();
      return int.tryParse(v?.toString() ?? '') ?? 0;
    }

    double asDouble(dynamic v) {
      if (v is double) return v;
      if (v is num) return v.toDouble();
      return double.tryParse(v?.toString() ?? '') ?? 0.0;
    }

    String asString(dynamic v) {
      if (v == null) return '';
      return v.toString();
    }

    return PdfQuestion(
      questionNumber: asInt(
        json['question_number'] ?? json['questionNumber'] ?? json['number'],
      ),
      questionText: asString(
          json['question_text'] ?? json['questionText'] ?? json['text']),
      parts: (json['parts'] as List?)
              ?.map((e) => PdfPart.fromJson(Map<String, dynamic>.from(e)))
              .toList() ??
          const [],
      yPosition: asDouble(json['y_position'] ?? json['yPosition'] ?? json['y']),
      xPosition: asDouble(json['x_position'] ?? json['xPosition'] ?? json['x']),
      width: asDouble(json['width']),
      height: asDouble(json['height']),
      pageNumber:
          asInt(json['page_number'] ?? json['pageNumber'] ?? json['page']),
      contextText: asString(json['context_text'] ?? json['contextText']),
      figurePaths: (json['figure_paths'] as List? ??
              json['figurePaths'] as List? ??
              const [])
          .map((e) => e.toString())
          .where((e) => e.isNotEmpty)
          .toList(),
      figureRegions: (json['figure_regions'] as List? ??
              json['figureRegions'] as List? ??
              const [])
          .whereType<Map>()
          .map((e) => PdfFigureRegion.fromJson(Map<String, dynamic>.from(e)))
          .toList(),
      figureBase64: json['figureBase64'] == null
          ? null
          : (json['figureBase64'] as List).cast<String>(),
      subjectTag: asString(
          json['subjectTag'] ?? json['subject_tag'] ?? json['subject']),
      chapterTag: asString(
          json['chapterTag'] ?? json['chapter_tag'] ?? json['chapter']),
      topicTag:
          asString(json['topicTag'] ?? json['topic_tag'] ?? json['topic']),
      difficultyTag: asString(json['difficultyTag'] ??
          json['difficulty_tag'] ??
          json['difficulty']),
      paperType: asString(
          json['paperType'] ?? json['paper_type'] ?? json['paperType']),
      boardTag:
          asString(json['boardTag'] ?? json['board_tag'] ?? json['board']),
      paperYear:
          asString(json['paperYear'] ?? json['paper_year'] ?? json['year']),
      spatialMetadata: Map<String, dynamic>.from(
        json['spatial_metadata'] as Map? ??
            json['spatialMetadata'] as Map? ??
            const {},
      ),
    );
  }

  Map<String, dynamic> toJson() => {
        'question_number': questionNumber,
        'question_text': questionText,
        'parts': parts.map((p) => p.toJson()).toList(),
        'y_position': yPosition,
        'x_position': xPosition,
        'width': width,
        'height': height,
        'page_number': pageNumber,
        'userAnswer': userAnswer,
        'correctAnswer': correctAnswer,
        'feedback': feedback,
        'isCorrect': isCorrect,
        'marksAvailable': marksAvailable,
        'marksAwarded': marksAwarded,
        'context_text': contextText,
        'figure_paths': figurePaths,
        'figure_regions':
            figureRegions.map((region) => region.toJson()).toList(),
        'subjectTag': subjectTag,
        'chapterTag': chapterTag,
        'topicTag': topicTag,
        'difficultyTag': difficultyTag,
        'paperType': paperType,
        'boardTag': boardTag,
        'paperYear': paperYear,
        'spatial_metadata': spatialMetadata,
      };
}

class PdfFigureRegion {
  final int pageNumber;
  final double x;
  final double top;
  final double width;
  final double height;

  const PdfFigureRegion({
    required this.pageNumber,
    required this.x,
    required this.top,
    required this.width,
    required this.height,
  });

  factory PdfFigureRegion.fromJson(Map<String, dynamic> json) {
    double asDouble(dynamic v) {
      if (v is double) return v;
      if (v is num) return v.toDouble();
      return double.tryParse(v?.toString() ?? '') ?? 0.0;
    }

    int asInt(dynamic v) {
      if (v is int) return v;
      if (v is num) return v.toInt();
      return int.tryParse(v?.toString() ?? '') ?? 0;
    }

    return PdfFigureRegion(
      pageNumber:
          asInt(json['page_number'] ?? json['pageNumber'] ?? json['page']),
      x: asDouble(json['x']),
      top: asDouble(json['top'] ?? json['y']),
      width: asDouble(json['width']),
      height: asDouble(json['height']),
    );
  }

  Map<String, dynamic> toJson() => {
        'page_number': pageNumber,
        'x': x,
        'top': top,
        'width': width,
        'height': height,
      };
}

class PdfPart {
  final String label;
  final String text;
  const PdfPart({required this.label, required this.text});

  factory PdfPart.fromJson(Map<String, dynamic> json) => PdfPart(
        label: (json['label'] ?? '').toString(),
        text: (json['text'] ?? '').toString(),
      );

  Map<String, dynamic> toJson() => {
        'label': label,
        'text': text,
      };
}

class ExamEvent {
  final String board;
  final String subject;
  final String label;
  final DateTime startDate;
  final DateTime endDate;
  final String source;
  final String? loadState;

  const ExamEvent({
    required this.board,
    required this.subject,
    required this.label,
    required this.startDate,
    required this.endDate,
    required this.source,
    this.loadState,
  });

  factory ExamEvent.fromJson(Map<String, dynamic> json) => ExamEvent(
        board: (json['board'] ?? '').toString(),
        subject: (json['subject'] ?? '').toString(),
        label: (json['label'] ?? '').toString(),
        startDate: DateTime.tryParse(
                (json['start_date'] ?? json['startDate'] ?? '').toString()) ??
            DateTime.now(),
        endDate: DateTime.tryParse(
                (json['end_date'] ?? json['endDate'] ?? '').toString()) ??
            DateTime.now(),
        source: (json['source'] ?? '').toString(),
        loadState:
            (json['load_state'] ?? json['loadState'] ?? 'balanced').toString(),
      );

  Map<String, dynamic> toJson() => {
        'board': board,
        'subject': subject,
        'label': label,
        'start_date': startDate.toIso8601String().split('T').first,
        'end_date': endDate.toIso8601String().split('T').first,
        'source': source,
        if (loadState != null) 'load_state': loadState,
      };
}
