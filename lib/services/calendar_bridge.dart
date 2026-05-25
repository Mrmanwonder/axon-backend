import 'package:googleapis/calendar/v3.dart' as google_api;
import 'package:syncfusion_flutter_calendar/calendar.dart';

import '../models/academic_engine_models.dart';
import 'backwards_induction_planner.dart';

typedef GoogleSyncIngestor = Future<void> Function(List<CalendarSyncEvent> events);
typedef PlannerRedistributor = Future<PlannerResult> Function(
  AppointmentDragUpdateDetails details,
);

class CalendarBridge extends CalendarDataSource
    implements SyncfusionCalendarBridge, GoogleCalendarBridge {
  CalendarBridge({
    this.googleCalendarApi,
    List<Appointment>? seedAppointments,
    this.onGoogleEventsIngested,
    this.onPlannerRedistributionRequested,
  }) {
    appointments = List<Appointment>.from(seedAppointments ?? const []);
  }

  final google_api.CalendarApi? googleCalendarApi;
  final GoogleSyncIngestor? onGoogleEventsIngested;
  final PlannerRedistributor? onPlannerRedistributionRequested;

  Future<void> syncFromGoogle() async {
    final api = googleCalendarApi;
    if (api == null) {
      return;
    }

    final feed = await api.events.list(
      'primary',
      singleEvents: true,
      orderBy: 'startTime',
    );

    final importedAppointments = <Appointment>[];
    final importedEvents = <CalendarSyncEvent>[];
    for (final event in feed.items ?? const <google_api.Event>[]) {
      final start = event.start?.dateTime?.toLocal();
      final end = event.end?.dateTime?.toLocal();
      if (start == null || end == null) {
        continue;
      }

      final axonEvent = CalendarSyncEvent(
        title: event.summary ?? 'Study Block',
        start: start,
        end: end,
        description: event.description ?? '',
      );
      importedEvents.add(axonEvent);
      importedAppointments.add(_toAppointment(axonEvent));
    }

    appointments = importedAppointments;
    notifyListeners(CalendarDataSourceAction.reset, appointments!);
    if (onGoogleEventsIngested != null && importedEvents.isNotEmpty) {
      await onGoogleEventsIngested!(importedEvents);
    }
  }

  Future<void> pushAxonTaskToGoogle(Appointment task) async {
    final api = googleCalendarApi;
    if (api == null) {
      return;
    }

    final googleEvent = google_api.Event()
      ..summary = '[Axon] ${task.subject.isNotEmpty ? task.subject : (task.notes ?? 'Study Session')}'
      ..description = task.notes ?? ''
      ..start = google_api.EventDateTime(dateTime: task.startTime.toUtc())
      ..end = google_api.EventDateTime(dateTime: task.endTime.toUtc());

    await api.events.insert(googleEvent, 'primary');
  }

  @override
  Future<void> syncStudyPlan(List<CalendarSyncEvent> events) async {
    final mappedAppointments = events.map(_toAppointment).toList();
    appointments = mappedAppointments;
    notifyListeners(CalendarDataSourceAction.reset, appointments!);

    for (final appointment in mappedAppointments) {
      await pushAxonTaskToGoogle(appointment);
    }
  }

  Future<void> handleAppointmentChange(
    AppointmentDragUpdateDetails details,
  ) async {
    final redistribution = onPlannerRedistributionRequested;
    if (redistribution == null) {
      return;
    }

    final nextPlan = await redistribution(details);
    await syncStudyPlan(
      nextPlan.days
          .where((day) => day.tasks.isNotEmpty)
          .map(
            (day) => CalendarSyncEvent(
              title: 'Axon Study Block',
              start: DateTime(day.date.year, day.date.month, day.date.day, 17),
              end: DateTime(day.date.year, day.date.month, day.date.day, 18),
              description: day.tasks
                  .map((task) => '${task.title} (${task.learningObjectiveId})')
                  .join(', '),
            ),
          )
          .toList(),
    );
  }

  Appointment _toAppointment(CalendarSyncEvent event) {
    return Appointment(
      startTime: event.start,
      endTime: event.end,
      subject: event.title,
      notes: event.description,
      isAllDay: false,
    );
  }
}
