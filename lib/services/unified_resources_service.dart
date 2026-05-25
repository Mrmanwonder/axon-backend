// lib/services/unified_resources_service.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';

enum ResourceSource { local, external }

class UnifiedResourcesService {
  static final UnifiedResourcesService _instance =
      UnifiedResourcesService._internal();
  factory UnifiedResourcesService() => _instance;
  UnifiedResourcesService._internal();

  // All available resources with external URLs (free, publicly accessible)
  static const Map<String, SubjectResourceConfig> resources = {
    // ============ IGCSE ============
    '0610': SubjectResourceConfig(
      code: '0610',
      name: 'Biology',
      level: 'IGCSE',
      chapters: [
        'Characteristics and Classification',
        'Organisation of the Organism',
        'Movement In and Out of Cells',
        'Biological Molecules',
        'Nutrition',
        'Respiration',
        'Transport',
        'Coordination and Response',
        'Reproduction',
        'Genetics',
        'Biotechnology',
        'Ecology and Environment',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes Biology',
            url: 'https://znotes.online/cie/igcse/biology/',
            source: 'znotes'),
        ResourceLink(
            name: 'SaveMyExams',
            url: 'https://www.savemyexams.com/igcse/biology/notes/',
            source: 'savemyexams'),
        ResourceLink(
            name: 'Khan Academy Biology',
            url: 'https://www.khanacademy.org/science/biology',
            source: 'khanacademy'),
        ResourceLink(
            name: 'Specimen Paper 1',
            url:
                'https://www.cambridgeinternational.org/Images/43201-specimen-paper-1.pdf',
            source: 'cambridge'),
      ],
    ),
    '0620': SubjectResourceConfig(
      code: '0620',
      name: 'Chemistry',
      level: 'IGCSE',
      chapters: [
        'States of Matter',
        'Atomic Structure',
        'Chemical Bonding',
        'Chemical Formulae',
        'The Periodic Table',
        'Electrolysis',
        'Acids, Bases and Salts',
        'Chemical Reactivity',
        'Metals',
        'Organic Chemistry',
        'Chemical Analysis',
      ],
      resources: [
        ResourceLink(
            name: 'ChemGuide',
            url: 'https://www.chemguide.co.uk/',
            source: 'chemguide'),
        ResourceLink(
            name: 'ZNotes Chemistry',
            url: 'https://znotes.online/cie/igcse/chemistry/',
            source: 'znotes'),
        ResourceLink(
            name: 'SaveMyExams',
            url: 'https://www.savemyexams.com/igcse/chemistry/notes/',
            source: 'savemyexams'),
      ],
    ),
    '0625': SubjectResourceConfig(
      code: '0625',
      name: 'Physics',
      level: 'IGCSE',
      chapters: [
        'Motion',
        'Force and Motion',
        'Forces and Pressure',
        'Work, Energy and Power',
        'Thermal Energy',
        'Waves',
        'Light and EM Spectrum',
        'Sound',
        'Magnetism and Electromagnetism',
        'Electric Circuits',
        'Electronics',
        'Radioactivity',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes Physics',
            url: 'https://znotes.online/cie/igcse/physics/',
            source: 'znotes'),
        ResourceLink(
            name: 'SaveMyExams',
            url: 'https://www.savemyexams.com/igcse/physics/notes/',
            source: 'savemyexams'),
        ResourceLink(
            name: 'Physics Tutor',
            url: 'https://www.youtube.com/c/PhysicsALevelTutorial',
            source: 'youtube'),
      ],
    ),
    '0580': SubjectResourceConfig(
      code: '0580',
      name: 'Mathematics',
      level: 'IGCSE',
      chapters: [
        'Number',
        'Algebra and Graphs',
        'Geometry',
        'Length and Area',
        'Trigonometry',
        'Vectors',
        'Probability',
        'Statistics',
      ],
      resources: [
        ResourceLink(
            name: 'Pixi Maths',
            url: 'https://www.piximaths.co.uk/',
            source: 'piximaths'),
        ResourceLink(
            name: 'ZNotes Maths',
            url: 'https://znotes.online/cie/igcse/mathematics/',
            source: 'znotes'),
        ResourceLink(
            name: 'Khan Academy Maths',
            url: 'https://www.khanacademy.org/math',
            source: 'khanacademy'),
      ],
    ),
    '0478': SubjectResourceConfig(
      code: '0478',
      name: 'Computer Science',
      level: 'IGCSE',
      chapters: [
        'Data Representation',
        'Communication',
        'Hardware',
        'Software',
        'Algorithm',
        'Program Design',
        'Coding',
        'Databases',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes CS',
            url: 'https://znotes.online/cie/igcse/computer-science/',
            source: 'znotes'),
      ],
    ),
    '0455': SubjectResourceConfig(
      code: '0455',
      name: 'Economics',
      level: 'IGCSE',
      chapters: [
        'Basic Economic Problem',
        'Allocation of Resources',
        'Microeconomics',
        'Macroeconomics',
        'Economic Development',
        'International Trade',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes Economics',
            url: 'https://znotes.online/cie/igcse/economics/',
            source: 'znotes'),
      ],
    ),
    // ============ AS & A LEVEL ============
    '9709': SubjectResourceConfig(
      code: '9709',
      name: 'Mathematics',
      level: 'AS & A Level',
      chapters: [
        'Quadratics',
        'Functions',
        'Coordinate Geometry',
        'Circular Measure',
        'Trigonometry',
        'Series',
        'Differentiation',
        'Integration',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes A-Level Maths',
            url: 'https://znotes.online/cie/as-a-level/mathematics/',
            source: 'znotes'),
        ResourceLink(
            name: 'SaveMyExams A-Level',
            url: 'https://www.savemyexams.com/alevel/maths/',
            source: 'savemyexams'),
      ],
    ),
    '9700': SubjectResourceConfig(
      code: '9700',
      name: 'Biology',
      level: 'AS & A Level',
      chapters: [
        'Cell Structure',
        'Biological Molecules',
        'Enzymes',
        'Cell Membranes',
        'Nucleic Acids',
        'Energy and Respiration',
        'Photosynthesis',
        'Homeostasis',
        'Inheritance',
        'Selection',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes A-Level Biology',
            url: 'https://znotes.online/cie/as-a-level/biology/',
            source: 'znotes'),
        ResourceLink(
            name: 'SaveMyExams',
            url: 'https://www.savemyexams.com/alevel/biology/',
            source: 'savemyexams'),
      ],
    ),
    '9701': SubjectResourceConfig(
      code: '9701',
      name: 'Chemistry',
      level: 'AS & A Level',
      chapters: [
        'Atomic Structure',
        'Chemical Bonding',
        'States of Matter',
        'Chemical Energetics',
        'Electrochemistry',
        'Equilibria',
        'Reaction Kinetics',
        'Inorganic Chemistry',
        'Organic Chemistry',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes A-Level Chemistry',
            url: 'https://znotes.online/cie/as-a-level/chemistry/',
            source: 'znotes'),
        ResourceLink(
            name: 'ChemGuide',
            url: 'https://www.chemguide.co.uk/',
            source: 'chemguide'),
      ],
    ),
    '9702': SubjectResourceConfig(
      code: '9702',
      name: 'Physics',
      level: 'AS & A Level',
      chapters: [
        'Kinematics',
        'Dynamics',
        'Forces',
        'Work and Energy',
        'Waves',
        'Electric Fields',
        'Capacitors',
        'Magnetic Fields',
        'Electromagnetic Induction',
        'Nuclear Physics',
      ],
      resources: [
        ResourceLink(
            name: 'ZNotes A-Level Physics',
            url: 'https://znotes.online/cie/as-a-level/physics/',
            source: 'znotes'),
        ResourceLink(
            name: 'Physics YouTube',
            url: 'https://www.youtube.com/c/PhysicsALevelTutorial',
            source: 'youtube'),
      ],
    ),
  };

  SubjectResourceConfig? getConfig(String code) => resources[code];

  List<String> getAllCodes() => resources.keys.toList();

  Future<void> openResource(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> openInAppViewer(String url, String title) async {
    // Uses media viewer service
    // Navigation handled by router
  }
}

class SubjectResourceConfig {
  final String code;
  final String name;
  final String level;
  final List<String> chapters;
  final List<ResourceLink> resources;

  const SubjectResourceConfig({
    required this.code,
    required this.name,
    required this.level,
    required this.chapters,
    required this.resources,
  });
}

class ResourceLink {
  final String name;
  final String url;
  final String source;

  const ResourceLink({
    required this.name,
    required this.url,
    required this.source,
  });
}

final unifiedResourcesServiceProvider =
    Provider<UnifiedResourcesService>((ref) {
  return UnifiedResourcesService();
});

final subjectResourceConfigProvider =
    Provider.family<SubjectResourceConfig?, String>((ref, code) {
  return UnifiedResourcesService().getConfig(code);
});
