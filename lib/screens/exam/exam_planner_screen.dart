// lib/screens/exam/exam_planner_screen.dart
// ─────────────────────────────────────────────────────────────────
// Full Exam Planner Screen
// Countdown, milestones, past papers, checklists, formulas
// ─────────────────────────────────────────────────────────────────

import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme/app_theme.dart';
import 'widgets/countdown_tab.dart';
import 'widgets/milestones_tab.dart';
import 'widgets/past_papers_tab.dart';
import 'widgets/checklists_tab.dart';
import 'widgets/formulas_tab.dart';

class ExamPlannerScreen extends ConsumerStatefulWidget {
  const ExamPlannerScreen({super.key});

  @override
  ConsumerState<ExamPlannerScreen> createState() => _ExamPlannerScreenState();
}

class _ExamPlannerScreenState extends ConsumerState<ExamPlannerScreen>
    with TickerProviderStateMixin {
  late TabController _tabController;
  late AnimationController _transitionController;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);

    _transitionController = AnimationController(
      duration: const Duration(milliseconds: 800),
      vsync: this,
    );

    _scaleAnimation = Tween<double>(begin: 0.8, end: 1.0).animate(
      CurvedAnimation(
          parent: _transitionController, curve: Curves.easeOutCubic),
    );

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _transitionController.forward();
      }
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    _transitionController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AxonColors.oxfordBlueDark,
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          Positioned(
            top: -100,
            right: -50,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AxonColors.accent.withValues(alpha: 0.15),
              ),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 80, sigmaY: 80),
                child: Container(),
              ),
            ),
          ),
          SafeArea(
            child: FadeTransition(
              opacity: _scaleAnimation,
              child: ScaleTransition(
                scale: _scaleAnimation,
                child: Column(
                  children: [
                    _buildExamPlannerHeader(),
                    Expanded(
                      child: TabBarView(
                        controller: _tabController,
                        children: const [
                          CountdownTab(),
                          MilestonesTab(),
                          PastPapersTab(),
                          ChecklistsTab(),
                          FormulasTab(),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExamPlannerHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Row(
        children: [
          IconButton(
            icon: const Icon(Icons.arrow_back_ios_new_rounded,
                color: Colors.white, size: 20),
            onPressed: () {
              if (Navigator.of(context).canPop()) {
                Navigator.of(context).pop();
              } else {
                // Navigate to home if can't pop
                context.go('/home');
              }
            },
          ),
          const SizedBox(width: 4),
          Expanded(
            child: Text(
              'EXAM PLANNER',
              style: GoogleFonts.googleSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.5,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
