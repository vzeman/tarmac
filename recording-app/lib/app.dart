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
      theme: _buildRoadSurveyTheme(Brightness.light),
      darkTheme: _buildRoadSurveyTheme(Brightness.dark),
      themeMode: _settings.displayTheme == DisplayTheme.night
          ? ThemeMode.dark
          : ThemeMode.light,
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

ThemeData _buildRoadSurveyTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final scheme =
      ColorScheme.fromSeed(
        seedColor: const Color(0xFF0E7A45),
        brightness: brightness,
      ).copyWith(
        primary: isDark ? const Color(0xFF4EE487) : const Color(0xFF006B3C),
        secondary: isDark ? const Color(0xFFFFC44D) : const Color(0xFF8A5A00),
        tertiary: isDark ? const Color(0xFF8FD5FF) : const Color(0xFF006A92),
        error: isDark ? const Color(0xFFFF8A80) : const Color(0xFFB00020),
        surface: isDark ? const Color(0xFF101418) : const Color(0xFFFCFCF8),
        surfaceContainerHighest: isDark
            ? const Color(0xFF20262C)
            : const Color(0xFFE7ECE5),
        outline: isDark ? const Color(0xFF8B949E) : const Color(0xFF59635D),
      );

  final base = ThemeData(
    colorScheme: scheme,
    brightness: brightness,
    useMaterial3: true,
    visualDensity: VisualDensity.standard,
    scaffoldBackgroundColor: isDark
        ? const Color(0xFF07090B)
        : const Color(0xFFF5F7F0),
  );

  final textTheme = base.textTheme.apply(
    bodyColor: scheme.onSurface,
    displayColor: scheme.onSurface,
  );

  return base.copyWith(
    textTheme: textTheme,
    appBarTheme: AppBarTheme(
      centerTitle: false,
      backgroundColor: scheme.surface,
      foregroundColor: scheme.onSurface,
      titleTextStyle: textTheme.titleLarge?.copyWith(
        fontWeight: FontWeight.w800,
      ),
    ),
    navigationRailTheme: NavigationRailThemeData(
      backgroundColor: scheme.surface,
      selectedIconTheme: IconThemeData(color: scheme.primary, size: 30),
      unselectedIconTheme: IconThemeData(color: scheme.onSurfaceVariant),
      selectedLabelTextStyle: textTheme.labelLarge?.copyWith(
        color: scheme.primary,
        fontWeight: FontWeight.w800,
      ),
      unselectedLabelTextStyle: textTheme.labelLarge,
      minWidth: 86,
      minExtendedWidth: 112,
    ),
    navigationBarTheme: NavigationBarThemeData(
      height: 72,
      backgroundColor: scheme.surface,
      indicatorColor: scheme.primaryContainer,
      labelTextStyle: WidgetStatePropertyAll(
        textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w700),
      ),
    ),
    listTileTheme: const ListTileThemeData(
      minTileHeight: 64,
      contentPadding: EdgeInsets.symmetric(horizontal: 18, vertical: 6),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: const Size(64, 64),
        textStyle: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(64, 64),
        textStyle: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        minimumSize: const Size(64, 64),
        textStyle: textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800),
      ),
    ),
    sliderTheme: base.sliderTheme.copyWith(
      trackHeight: 6,
      thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 12),
    ),
    switchTheme: SwitchThemeData(
      thumbColor: WidgetStateProperty.resolveWith((states) {
        return states.contains(WidgetState.selected)
            ? scheme.primary
            : scheme.outline;
      }),
    ),
    cardTheme: CardThemeData(
      color: scheme.surface,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(8),
        side: BorderSide(color: scheme.outlineVariant),
      ),
    ),
  );
}
