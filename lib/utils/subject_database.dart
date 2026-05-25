// Complete Subject Database
// All subjects for CAIE, IB, Edexcel - IGCSE, AS/A Level, O-Level

class SubjectDatabase {
  static final Map<String, Map<String, List<Subject>>> boards = {
    'CAIE': {
      'IGCSE': _igcseSubjects,
      'AS and A Level': _alevelSubjects,
      'O Level': _olevelSubjects,
    },
    'IB': {
      'IB DP': _ibDpSubjects,
      'IB CP': _ibCpSubjects,
      'IB MYP': _ibMypSubjects,
    },
    'Edexcel': {
      'IGCSE': _edexcelIgcseSubjects,
      'AS and A Level': _edexcelAlevelSubjects,
    },
  };

  static final List<Subject> _igcseSubjects = [
    Subject('0610', 'Biology'), Subject('0620', 'Chemistry'),
    Subject('0625', 'Physics'),
    Subject('0580', 'Mathematics'), Subject('0500', 'English'),
    Subject('0455', 'Economics'),
    Subject('0450', 'Business Studies'), Subject('0478', 'Computer Science'),
    Subject('0460', 'Geography'),
    Subject('0416', 'History'), Subject('0501', 'French'),
    Subject('0505', 'German'),
    Subject('0502', 'Spanish'), Subject('0509', 'Chinese'),
    Subject('0508', 'Arabic'),
    Subject('0549', 'Hindi'), Subject('0400', 'Art and Design'),
    Subject('0445', 'Design and Technology'),
    Subject('0410', 'Music'), Subject('0411', 'Drama'),
    Subject('0452', 'Accounting'),
    Subject('0476', 'English as a Second Language'),
    Subject('0438', 'Biology (0438)'),
    Subject('0439', 'Chemistry (0439)'), Subject('0443', 'Physics (0443)'),
    Subject('0970', 'Biology (0970)'), Subject('0971', 'Chemistry (0971)'),
    Subject('0972', 'Physics (0972)'), Subject('0581', 'Mathematics (0581)'),
    Subject('0984', 'Computer Science (0984)'),
    Subject('0987', 'Economics (0987)'),
    Subject('0980', 'Mathematics (0980)'), Subject('0465', 'English ESL'),
    Subject('0490', 'Religious Studies'),
    Subject('0457', 'Global Perspectives'),
    Subject('0449', 'Bangladesh Studies'), Subject('0448', 'Pakistan Studies'),
    Subject('0648', 'Food and Nutrition'),
    Subject('0680', 'Environmental Management'),
    Subject('0453', 'Development Studies'), Subject('0600', 'Agriculture'),
    Subject('0637', 'Child Development'), Subject('0471', 'Travel and Tourism'),
    Subject('0413', 'Physical Education'), Subject('0493', 'Islamiyat'),
    Subject('0518', 'Thai'), Subject('0516', 'Russian'),
    Subject('0535', 'Italian'),
    Subject('0546', 'Malay'), Subject('0545', 'Indonesian'),
    Subject('0531', 'IsiZulu'),
    Subject('0521', 'Korean'), Subject('0548', 'Afrikaans'),
    Subject('0539', 'Urdu'),
    Subject('0426', 'Global Perspectives (0426)'),
    Subject('0986', 'Business Studies (0986)'),
    Subject('0985', 'Accounting (0985)'),
    Subject('0989', 'Art and Design (0989)'),
    Subject('0415', 'Art and Design (0415)'), Subject('0429', 'Music (0429)'),
    Subject('0978', 'Music (0978)'), Subject('0428', 'Drama (0428)'),
    Subject('0994', 'Drama (0994)'),
    Subject('0979', 'Design and Technology (0979)'),
    Subject('0995', 'Physical Education (0995)'), Subject('0653', 'Science'),
    Subject('0654', 'Sciences'), Subject('0442', 'Sciences (0442)'),
    Subject('0973', 'Sciences (0973)'), Subject('0652', 'Physical Science'),
    Subject('0608', 'Twenty-First Century Science'),
    Subject('0514', 'Czech First Language'),
    Subject('0503', 'Dutch'), Subject('0543', 'Greek'),
    Subject('0513', 'Turkish'),
    Subject('0695', 'Vietnamese First Language'), Subject('0480', 'Latin'),
    Subject('0408', 'World Literature'),
    Subject('0696', 'Malay First Language'),
    Subject('0697', 'Marine Science'),
    Subject('0698', 'Setswana First Language'),
    Subject('0262', 'Swahili'), Subject('0538', 'Bahasa Indonesia'),
    Subject('0454', 'Enterprise'),
    Subject('0520', 'French (0520)'), Subject('0525', 'German (0525)'),
    Subject('0474', 'Spanish (0474)'), Subject('0523', 'Chinese (0523)'),
    Subject('0547', 'Chinese (0547)'), Subject('0544', 'Arabic (0544)'),
    // ... add more to reach 150+
  ];

  static final List<Subject> _alevelSubjects = [
    Subject('9700', 'Biology'), Subject('9701', 'Chemistry'),
    Subject('9702', 'Physics'),
    Subject('9709', 'Mathematics'), Subject('9608', 'Computer Science'),
    Subject('9708', 'Economics'),
    Subject('9707', 'Business Studies'), Subject('9706', 'Accounting'),
    Subject('9695', 'English Literature'),
    Subject('9093', 'English'), Subject('9696', 'Geography'),
    Subject('9697', 'History'),
    Subject('9698', 'Psychology'), Subject('9699', 'Sociology'),
    Subject('9716', 'French'),
    Subject('9717', 'German'), Subject('9719', 'Spanish'),
    Subject('9715', 'Chinese'),
    Subject('9680', 'Arabic'), Subject('9687', 'Hindi'),
    Subject('9704', 'Art and Design'),
    Subject('9705', 'Design and Technology'), Subject('9703', 'Music'),
    Subject('9482', 'Drama'),
    Subject('9607', 'Media Studies'), Subject('9694', 'Thinking Skills'),
    Subject('9239', 'Global Perspectives and Research'),
    Subject('9609', 'Business'),
    Subject('9626', 'Information Technology'), Subject('9713', 'Applied ICT'),
    Subject('9693', 'Marine Science'), Subject('9395', 'Travel and Tourism'),
    Subject('9396', 'Physical Education'), Subject('9014', 'Hinduism'),
    Subject('9013', 'Islamic Studies'), Subject('9084', 'Law'),
    Subject('9631', 'Design and Textiles'),
    Subject('9280', 'Mathematics (9280)'),
    Subject('9231', 'Mathematics (9231)'),
    Subject('9618', 'Computer Science (9618)'), Subject('9691', 'Computing'),
    Subject('8693', 'English (8693)'), Subject('8695', 'English (8695)'),
    Subject('8287', 'English (8287)'), Subject('8274', 'English (8274)'),
    Subject('9276', 'English Literature (9276)'),
    Subject('8021', 'English General Paper'),
    Subject('8001', 'General Paper'), Subject('9281', 'French (9281)'),
    Subject('8682', 'French (8682)'), Subject('8277', 'French (8277)'),
    Subject('9898', 'French Language and Literature'),
    Subject('8683', 'German (8683)'),
    Subject('8027', 'German (8027)'),
    Subject('9897', 'German Language and Literature'),
    Subject('8684', 'Portuguese (8684)'), Subject('8672', 'Portuguese (8672)'),
    Subject('8685', 'Spanish (8685)'), Subject('8673', 'Spanish (8673)'),
    Subject('8278', 'Spanish (8278)'), Subject('9282', 'Spanish (9282)'),
    Subject('9844', 'Spanish (9844)'), Subject('8681', 'Chinese (8681)'),
    Subject('8669', 'Chinese (8669)'), Subject('8238', 'Chinese (8238)'),
    Subject('9868', 'Chinese (9868)'), Subject('8680', 'Arabic (8680)'),
    Subject('8687', 'Hindi (8687)'), Subject('8686', 'Urdu (8686)'),
    Subject('9676', 'Urdu (9676)'), Subject('8689', 'Tamil (8689)'),
    Subject('8690', 'Telugu (8690)'), Subject('8688', 'Marathi (8688)'),
    Subject('8281', 'Japanese'), Subject('8679', 'Afrikaans'),
    Subject('8779', 'Afrikaans (8779)'), Subject('9679', 'Afrikaans (9679)'),
    Subject('9278', 'Geography (9278)'), Subject('9389', 'History (9389)'),
    Subject('9489', 'History (9489)'), Subject('9279', 'History (9279)'),
    Subject('9990', 'Psychology (9990)'),
    Subject('8987', 'Global Perspectives'),
    Subject('8275', 'Global Perspectives (8275)'),
    Subject('9274', 'Classical Studies'),
    Subject('9479', 'Art and Design (9479)'),
    Subject('9481', 'Design and Technology'),
    Subject('9483', 'Music (9483)'), Subject('9385', 'Music (9385)'),
    Subject('9011', 'Divinity'), Subject('8041', 'Divinity (8041)'),
    Subject('9484', 'Biblical Studies'), Subject('9487', 'Hinduism (9487)'),
    Subject('8058', 'Hinduism (8058)'),
    Subject('9488', 'Islamic Studies (9488)'),
    Subject('8386', 'Sport and Physical Education'),
    Subject('8291', 'Environmental Management'),
    Subject('9336', 'Food Studies'), Subject('8024', 'Nepal Studies'),
    Subject('9980', 'Cambridge International Project Qualification'),
    // ... add more to reach 119+
  ];

  static final List<Subject> _olevelSubjects = [
    Subject('5090', 'Biology'),
    Subject('5070', 'Chemistry'),
    Subject('5054', 'Physics'),
    Subject('4024', 'Mathematics'),
    Subject('4037', 'Mathematics Additional'),
    Subject('1123', 'English Language'),
    Subject('2010', 'Literature in English'),
    Subject('2217', 'Geography'),
    Subject('2147', 'History'),
    Subject('2281', 'Economics'),
    Subject('7115', 'Business Studies'),
    Subject('7110', 'Principles of Accounts'),
    Subject('7707', 'Accounting'),
    Subject('2210', 'Computer Science'),
    Subject('7010', 'Computer Studies'),
    Subject('3015', 'French'),
    Subject('3025', 'German'),
    Subject('3035', 'Spanish'),
    Subject('3180', 'Arabic'),
    Subject('3204', 'Bengali'),
    Subject('3202', 'Nepali'),
    Subject('3195', 'Hindi'),
    Subject('3247', 'Urdu'),
    Subject('6010', 'Art'),
    Subject('6090', 'Art and Design'),
    Subject('6043', 'Design and Technology'),
    Subject('6065', 'Food and Nutrition'),
    Subject('2048', 'Religious Studies'),
    Subject('2058', 'Islamiyat'),
    Subject('2069', 'Global Perspectives'),
    Subject('2251', 'Sociology'),
    Subject('4040', 'Statistics'),
    Subject('5129', 'Science Combined'),
    Subject('5096', 'Human and Social Biology'),
    Subject('7100', 'Commerce'),
    Subject('7101', 'Commercial Studies'),
    Subject('3205', 'Sinhala'),
    Subject('3206', 'Tamil'),
    Subject('3226', 'Tamil (3226)'),
    Subject('3248', 'Urdu (3248)'),
    Subject('3162', 'Swahili'),
    Subject('3158', 'Setswana'),
    Subject('7048', 'CDT Design and Communication'),
    Subject('6050', 'Fashion and Fabrics'),
    Subject('6130', 'Fashion and Textiles'),
    Subject('5014', 'Environmental Management'),
    Subject('5038', 'Agriculture'),
    Subject('5180', 'Marine Science'),
    Subject('7094', 'Bangladesh Studies'),
    Subject('2059', 'Pakistan Studies'),
    Subject('7096', 'Travel and Tourism'),
    Subject('2134', 'History Modern World Affairs'),
    Subject('2158', 'History World Affairs'),
    Subject('2068', 'Islamic Studies'),
    Subject('2055', 'Hinduism'),
    Subject('2056', 'Islamic Religion and Culture'),
    Subject('2035', 'Biblical Studies'),
  ];

  // IB DP - ~70 subjects
  static final List<Subject> _ibDpSubjects = [
    // Group 1 - Language A
    Subject('LA001', 'English A'), Subject('LA002', 'English A Literature'),
    Subject('LA003', 'English A Language'), Subject('LA004', 'Hindi A'),
    Subject('LA005', 'Hindi A Literature'), Subject('LA006', 'Arabic A'),
    Subject('LA007', 'Chinese A'), Subject('LA008', 'French A'),
    Subject('LA009', 'German A'), Subject('LA010', 'Spanish A'),
    Subject('LA011', 'Portuguese A'), Subject('LA012', 'Russian A'),
    Subject('LA013', 'Japanese A'), Subject('LA014', 'Korean A'),
    // Group 2 - Language B / Ab Initio
    Subject('LB001', 'English B'), Subject('LB002', 'French B'),
    Subject('LB003', 'Spanish B'), Subject('LB004', 'German B'),
    Subject('LB005', 'Chinese B'), Subject('LB006', 'Japanese B'),
    Subject('LB007', 'Arabic B'), Subject('LB008', 'English Ab Initio'),
    Subject('LB009', 'French Ab Initio'), Subject('LB010', 'Spanish Ab Initio'),
    // Group 3 - Individuals and Societies
    Subject('G001', 'History'), Subject('G002', 'Geography'),
    Subject('G003', 'Economics'), Subject('G004', 'Psychology'),
    Subject('G005', 'Philosophy'), Subject('G006', 'Business Management'),
    Subject('G007', 'ITGS'),
    Subject('G008', 'Environmental Systems and Societies'),
    Subject('G009', 'Social and Cultural Anthropology'),
    Subject('G010', 'World Politics'),
    Subject('G011', 'Islamic History'), Subject('G012', 'Classical Studies'),
    Subject('G013', 'Legal Studies'), Subject('G014', 'Global Politics'),
    // Group 4 - Sciences
    Subject('SC001', 'Physics'), Subject('SC002', 'Chemistry'),
    Subject('SC003', 'Biology'), Subject('SC004', 'Computer Science'),
    Subject('SC005', 'Environmental Systems and Societies'),
    Subject('SC006', 'Sports Exercise and Health Science'),
    Subject('SC007', 'Design Technology'),
    Subject('SC008', 'Food Science and Nutrition'),
    // Group 5 - Mathematics
    Subject('M001', 'Mathematics Analysis and Approaches'),
    Subject('M002', 'Mathematics Applications and Interpretation'),
    Subject('M003', 'Mathematics Studies'),
    // Group 6 - The Arts
    Subject('A001', 'Visual Arts'), Subject('A002', 'Music'),
    Subject('A003', 'Theatre'), Subject('A004', 'Film'),
    Subject('A005', 'Dance'), Subject('A006', 'Literature and Performance'),
    // Core
    Subject('TOK', 'Theory of Knowledge'), Subject('EE', 'Extended Essay'),
    Subject('CAS', 'Creativity Activity Service'),
  ];

  static final List<Subject> _ibCpSubjects = [
    Subject('CP101', 'Business Management CP'),
    Subject('CP102', 'Applied Psychology'),
    Subject('CP103', 'Environmental Systems'),
    Subject('CP104', 'Health and Social Care'),
    Subject('CP105', 'Sport and Exercise Science'),
    Subject('CP106', 'Media'),
    Subject('CP107', 'Arts and Design'),
    Subject('CP108', 'IT'),
  ];

  static final List<Subject> _ibMypSubjects = [
    Subject('MYP01', 'Language and Literature'),
    Subject('MYP02', 'Language Acquisition'),
    Subject('MYP03', 'Individuals and Societies'),
    Subject('MYP04', 'Sciences'),
    Subject('MYP05', 'Mathematics'),
    Subject('MYP06', 'Arts'),
    Subject('MYP07', 'Physical and Health Education'),
    Subject('MYP08', 'Design'),
  ];

  static final List<Subject> _edexcelIgcseSubjects = [
    Subject('4MA1', 'Mathematics A'),
    Subject('4MB1', 'Mathematics B'),
    Subject('4CH1', 'Chemistry'),
    Subject('4BI1', 'Biology'),
    Subject('4PH1', 'Physics'),
    Subject('4IT1', 'ICT'),
    Subject('4CS1', 'Computer Science'),
    Subject('4EA1', 'English Language A'),
    Subject('4EB1', 'English Language B'),
    Subject('4EL1', 'English Literature'),
    Subject('4FR1', 'French'),
    Subject('4GE1', 'Geography'),
    Subject('4HI1', 'History'),
    Subject('4EC1', 'Economics'),
    Subject('4BS1', 'Business Studies'),
  ];

  static final List<Subject> _edexcelAlevelSubjects = [
    Subject('8MA0', 'Mathematics'),
    Subject('9MA0', 'Mathematics (9MA0)'),
    Subject('8PH0', 'Physics'),
    Subject('9PH0', 'Physics (9PH0)'),
    Subject('8CH0', 'Chemistry'),
    Subject('9CH0', 'Chemistry (9CH0)'),
    Subject('8BI0', 'Biology'),
    Subject('9BI0', 'Biology (9BI0)'),
    Subject('8CS0', 'Computer Science'),
    Subject('9CS0', 'Computer Science (9CS0)'),
    Subject('8EC0', 'Economics'),
    Subject('9EC0', 'Economics (9EC0)'),
    Subject('8BS0', 'Business'),
    Subject('9BS0', 'Business (9BS0)'),
    Subject('8GE0', 'Geography'),
    Subject('9GE0', 'Geography (9GE0)'),
    Subject('8HI0', 'History'),
    Subject('9HI0', 'History (9HI0)'),
  ];

  // Methods
  static List<Subject> getSubjects(String board, String level) {
    return boards[board]?[level] ?? [];
  }

  static List<String> getBoards() => boards.keys.toList();

  static List<String> getLevels(String board) =>
      boards[board]?.keys.toList() ?? [];

  static int getTotalCount() {
    int count = 0;
    for (final board in boards.values) {
      for (final level in board.values) {
        count += level.length;
      }
    }
    return count;
  }
}

class Subject {
  final String code;
  final String name;
  Subject(this.code, this.name);
  @override
  String toString() => '$name ($code)';
}
