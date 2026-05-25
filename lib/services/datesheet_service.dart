import 'exam_data_service.dart' as exam_data;

class DatesheetService {
  final exam_data.ExamDataService _delegate = exam_data.ExamDataService();

  List<DatesheetExamEvent> get allExams {
    return _delegate.allExams.map(DatesheetExamEvent.fromExamData).toList();
  }
}

class DatesheetExamEvent {
  final String code;
  final DateTime dateTime;

  const DatesheetExamEvent({
    required this.code,
    required this.dateTime,
  });

  factory DatesheetExamEvent.fromExamData(exam_data.ExamEvent event) {
    return DatesheetExamEvent(
      code: event.component,
      dateTime: event.date,
    );
  }
}
