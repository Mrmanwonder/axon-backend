import '../models/admissions_models.dart';

class MilestoneFactory {
  static List<AdmissionsMilestone> generateForTarget({
    required AdmissionsTarget target,
    DateTime? anchorDate,
  }) {
    final anchor = anchorDate ?? DateTime.now();
    final milestones = <AdmissionsMilestone>[];

    // Core Verification
    milestones.add(AdmissionsMilestone(
      id: '${target.id}_verify',
      targetId: target.id,
      title: 'Verify entry requirements for ${target.universityName}',
      dueAt: anchor.add(const Duration(days: 3)),
      completed: false,
      dependencyType: 'verification',
      phase: 'research',
      status: 'pending',
    ));

    // Determine timeline based on system and course
    final system = target.applicationSystem.toUpperCase();
    final courseLower = target.courseName.toLowerCase();

    if (system.contains('UCAS')) {
      _addUcasMilestones(milestones, target, anchor, courseLower);
    } else if (system.contains('DIRECT')) {
      _addDirectMilestones(milestones, target, anchor, courseLower);
    } else {
      _addGenericMilestones(milestones, target, anchor);
    }

    // STEM Admissions Tests (STEP, MAT, TMUA, CSAT)
    if (system.contains('UCAS') &&
        (courseLower.contains('math') || courseLower.contains('computer') || courseLower.contains('physics') || courseLower.contains('engineering'))) {
      if (target.universityName.toLowerCase().contains('cambridge') || 
          target.universityName.toLowerCase().contains('imperial') ||
          target.universityName.toLowerCase().contains('ucl') ||
          target.universityName.toLowerCase().contains('oxford')) {
        milestones.add(AdmissionsMilestone(
          id: '${target.id}_stem_test',
          targetId: target.id,
          title: 'Register and prepare for STEM Admissions Test',
          dueAt: anchor.add(const Duration(days: 45)),
          completed: false,
          dependencyType: 'examination',
          phase: 'testing',
          dependsOnIds: ['${target.id}_verify'],
          status: 'pending',
        ));
      }
    }

    return milestones;
  }

  static void _addUcasMilestones(
    List<AdmissionsMilestone> list,
    AdmissionsTarget target,
    DateTime anchor,
    String courseLower,
  ) {
    // Personal Statement
    list.add(AdmissionsMilestone(
      id: '${target.id}_evidence',
      targetId: target.id,
      title: 'Draft UCAS Personal Statement',
      dueAt: anchor.add(const Duration(days: 14)),
      completed: false,
      dependencyType: 'evidence',
      phase: 'portfolio',
      dependsOnIds: ['${target.id}_verify'],
      status: 'pending',
    ));

    // Submission
    final isOxbridge = target.universityName.toLowerCase().contains('oxford') || 
                       target.universityName.toLowerCase().contains('cambridge');
    final isMed = courseLower.contains('medicine') || courseLower.contains('dentistry');

    // Default UCAS deadline is typically Jan 31st of the target year.
    // Oxbridge/Med is Oct 15th of the prior year.
    // Just using the provided deadlineAt or an offset.
    final deadline = target.deadlineAt ?? anchor.add(Duration(days: (isOxbridge || isMed) ? 30 : 60));

    list.add(AdmissionsMilestone(
      id: '${target.id}_apply',
      targetId: target.id,
      title: 'Submit UCAS Application',
      dueAt: deadline,
      completed: false,
      dependencyType: 'submission',
      phase: 'application',
      dependsOnIds: ['${target.id}_verify', '${target.id}_evidence'],
      status: 'pending',
    ));
  }

  static void _addDirectMilestones(
    List<AdmissionsMilestone> list,
    AdmissionsTarget target,
    DateTime anchor,
    String courseLower,
  ) {
    list.add(AdmissionsMilestone(
      id: '${target.id}_evidence',
      targetId: target.id,
      title: 'Compile direct entry portfolio and transcripts',
      dueAt: anchor.add(const Duration(days: 21)),
      completed: false,
      dependencyType: 'evidence',
      phase: 'portfolio',
      dependsOnIds: ['${target.id}_verify'],
      status: 'pending',
    ));

    final deadline = target.deadlineAt ?? anchor.add(const Duration(days: 40));
    list.add(AdmissionsMilestone(
      id: '${target.id}_apply',
      targetId: target.id,
      title: 'Submit Direct Application to ${target.universityName}',
      dueAt: deadline,
      completed: false,
      dependencyType: 'submission',
      phase: 'application',
      dependsOnIds: ['${target.id}_verify', '${target.id}_evidence'],
      status: 'pending',
    ));
  }

  static void _addGenericMilestones(
    List<AdmissionsMilestone> list,
    AdmissionsTarget target,
    DateTime anchor,
  ) {
    list.add(AdmissionsMilestone(
      id: '${target.id}_evidence',
      targetId: target.id,
      title: 'Prepare application documents for ${target.universityName}',
      dueAt: anchor.add(const Duration(days: 14)),
      completed: false,
      dependencyType: 'evidence',
      phase: 'portfolio',
      dependsOnIds: ['${target.id}_verify'],
      status: 'pending',
    ));

    list.add(AdmissionsMilestone(
      id: '${target.id}_apply',
      targetId: target.id,
      title: 'Submit application to ${target.universityName}',
      dueAt: target.deadlineAt ?? anchor.add(const Duration(days: 30)),
      completed: false,
      dependencyType: 'submission',
      phase: 'application',
      dependsOnIds: ['${target.id}_verify', '${target.id}_evidence'],
      status: 'pending',
    ));
  }
}
