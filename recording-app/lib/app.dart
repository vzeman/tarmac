import 'package:flutter/material.dart';

import 'models/session_summary.dart';
import 'screens/record_screen.dart';
import 'screens/sessions_screen.dart';
import 'screens/settings_screen.dart';
import 'services/session_repository.dart';
import 'settings/app_settings.dart';
import 'settings/settings_store.dart';

class RoadSurveyApp extends StatefulWidget {
  const RoadSurveyApp({super.key});

  @override
  State<RoadSurveyApp> createState() => _RoadSurveyAppState();
}

class _RoadSurveyAppState extends State<RoadSurveyApp> {
  final SettingsStore _settingsStore = SettingsStore();
  final SessionRepository _sessionRepository = SessionRepository();

  late final Future<void> _bootstrapFuture;
  AppSettings _settings = AppSettings.defaults();
  List<SessionSummary> _sessions = [];
  int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    _bootstrapFuture = _bootstrap();
  }

  Future<void> _bootstrap() async {
    final settings = await _settingsStore.load();
    final sessions = await _sessionRepository.listSessions();
    if (!mounted) {
      return;
    }
    setState(() {
      _settings = settings;
      _sessions = sessions;
    });
  }

  Future<void> _saveSettings(AppSettings settings) async {
    setState(() => _settings = settings);
    await _settingsStore.save(settings);
  }

  Future<void> _reloadSessions() async {
    final sessions = await _sessionRepository.listSessions();
    if (!mounted) {
      return;
    }
    setState(() => _sessions = sessions);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RoadSurvey Recorder',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF216E5B),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: FutureBuilder<void>(
        future: _bootstrapFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Scaffold(
              body: Center(child: CircularProgressIndicator()),
            );
          }
          final pages = [
            RecordScreen(
              settings: _settings,
              sessionRepository: _sessionRepository,
              onSessionSaved: _reloadSessions,
            ),
            SessionsScreen(
              sessions: _sessions,
              sessionRepository: _sessionRepository,
              onRefresh: _reloadSessions,
            ),
            SettingsScreen(settings: _settings, onChanged: _saveSettings),
          ];
          return LayoutBuilder(
            builder: (context, constraints) {
              final isLandscape = constraints.maxWidth > constraints.maxHeight;
              if (isLandscape) {
                return Scaffold(
                  body: SafeArea(
                    child: Row(
                      children: [
                        NavigationRail(
                          selectedIndex: _selectedIndex,
                          onDestinationSelected: (index) {
                            setState(() => _selectedIndex = index);
                          },
                          labelType: NavigationRailLabelType.all,
                          destinations: const [
                            NavigationRailDestination(
                              icon: Icon(Icons.videocam_outlined),
                              selectedIcon: Icon(Icons.videocam),
                              label: Text('Record'),
                            ),
                            NavigationRailDestination(
                              icon: Icon(Icons.folder_outlined),
                              selectedIcon: Icon(Icons.folder),
                              label: Text('Sessions'),
                            ),
                            NavigationRailDestination(
                              icon: Icon(Icons.tune_outlined),
                              selectedIcon: Icon(Icons.tune),
                              label: Text('Settings'),
                            ),
                          ],
                        ),
                        const VerticalDivider(width: 1),
                        Expanded(
                          child: IndexedStack(
                            index: _selectedIndex,
                            children: pages,
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }

              return Scaffold(
                appBar: AppBar(title: Text(_titleFor(_selectedIndex))),
                body: IndexedStack(index: _selectedIndex, children: pages),
                bottomNavigationBar: NavigationBar(
                  selectedIndex: _selectedIndex,
                  onDestinationSelected: (index) {
                    setState(() => _selectedIndex = index);
                  },
                  destinations: const [
                    NavigationDestination(
                      icon: Icon(Icons.videocam_outlined),
                      selectedIcon: Icon(Icons.videocam),
                      label: 'Record',
                    ),
                    NavigationDestination(
                      icon: Icon(Icons.folder_outlined),
                      selectedIcon: Icon(Icons.folder),
                      label: 'Sessions',
                    ),
                    NavigationDestination(
                      icon: Icon(Icons.tune_outlined),
                      selectedIcon: Icon(Icons.tune),
                      label: 'Settings',
                    ),
                  ],
                ),
              );
            },
          );
        },
      ),
    );
  }

  String _titleFor(int index) {
    switch (index) {
      case 0:
        return 'RoadSurvey Recorder';
      case 1:
        return 'Sessions';
      case 2:
        return 'Settings';
      default:
        return 'RoadSurvey Recorder';
    }
  }
}
