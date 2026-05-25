import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AxonCard extends StatelessWidget {
  final Widget child;
  final EdgeInsets? padding;
  final EdgeInsets? margin;
  final double borderRadius;
  final Color? color;
  final VoidCallback? onTap;
  final double? width;
  final double? height;
  final Color? borderColor;
  final bool hasBorder;
  final Color? glowColor;
  final double? glowIntensity;
  final bool fullWidth;

  const AxonCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.borderRadius = 16,
    this.color,
    this.onTap,
    this.width,
    this.height,
    this.borderColor,
    this.hasBorder = false,
    this.glowColor,
    this.glowIntensity,
    this.fullWidth = false,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final effectiveBorderColor =
        borderColor ?? (isDark ? Colors.white12 : Colors.black12);
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: fullWidth ? double.infinity : width,
        height: height,
        margin: margin,
        padding: padding ?? const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color ?? (isDark ? const Color(0xFF1E1E2E) : Colors.white),
          borderRadius: BorderRadius.circular(borderRadius),
          border: hasBorder ? Border.all(color: effectiveBorderColor) : null,
          boxShadow: glowColor != null
              ? [
                  BoxShadow(
                      color: glowColor!.withValues(alpha: glowIntensity ?? 0.2),
                      blurRadius: 20)
                ]
              : null,
        ),
        child: child,
      ),
    );
  }
}

class AxonGlass extends StatelessWidget {
  final Widget child;
  final double borderRadius;
  final double blur;
  final EdgeInsets? padding;
  final Color? glowColor;
  final Color? backgroundColor;
  final double? glowIntensity;

  const AxonGlass({
    super.key,
    required this.child,
    this.borderRadius = 24,
    this.blur = 15,
    this.padding,
    this.glowColor,
    this.backgroundColor,
    this.glowIntensity,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: padding ?? const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: backgroundColor ??
            (isDark ? Colors.white.withValues(alpha: 0.05) : Colors.white),
        borderRadius: BorderRadius.circular(borderRadius),
        border: Border.all(
          color: isDark ? Colors.white12 : Colors.black12,
        ),
        boxShadow: glowColor != null
            ? [
                BoxShadow(
                    color: glowColor!.withValues(alpha: 0.2), blurRadius: 20)
              ]
            : null,
      ),
      child: child,
    );
  }
}

class DeltaBadge extends StatelessWidget {
  final String value;
  final bool isPositive;
  final String? label;
  final dynamic delta;

  const DeltaBadge({
    super.key,
    this.value = '',
    this.isPositive = true,
    this.label,
    this.delta,
  });

  @override
  Widget build(BuildContext context) {
    final badgeColor = isPositive ? Colors.green : Colors.red;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: badgeColor.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: badgeColor.withValues(alpha: 0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(isPositive ? Icons.arrow_upward : Icons.arrow_downward,
              color: badgeColor, size: 12),
          const SizedBox(width: 4),
          Text(value,
              style: TextStyle(
                  color: badgeColor,
                  fontSize: 12,
                  fontWeight: FontWeight.bold)),
          if (label != null) ...[
            const SizedBox(width: 4),
            Text(label!,
                style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.6), fontSize: 10)),
          ],
        ],
      ),
    );
  }
}

class AxonSpatialSlider extends StatelessWidget {
  final String label;
  final double value;
  final String Function(double) valueFormatter;
  final ValueChanged<double> onChanged;
  final Color? color;

  const AxonSpatialSlider({
    super.key,
    required this.label,
    required this.value,
    required this.valueFormatter,
    required this.onChanged,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: GoogleFonts.inter(
                color: isDark ? Colors.white70 : Colors.black87,
                fontSize: 14,
                fontWeight: FontWeight.w500)),
        const SizedBox(height: 8),
        SliderTheme(
          data: SliderThemeData(
            activeTrackColor: const Color(0xFF3A86FF),
            inactiveTrackColor: isDark ? Colors.white12 : Colors.black12,
            thumbColor: const Color(0xFF3A86FF),
            overlayColor: const Color(0xFF3A86FF).withValues(alpha: 0.2),
          ),
          child: Slider(value: value, onChanged: onChanged),
        ),
        Center(
          child: Text(valueFormatter(value),
              style: GoogleFonts.inter(
                  color: const Color(0xFF3A86FF),
                  fontSize: 16,
                  fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }
}

class CyberButton extends StatelessWidget {
  final String label;
  final VoidCallback? onTap;
  final bool isActive;
  final IconData? icon;
  final bool fullWidth;
  final bool isLoading;
  final Color? color;
  final Color? borderColor;
  final Color? textColor;
  final bool showShadow;

  const CyberButton({
    super.key,
    required this.label,
    this.onTap,
    this.isActive = false,
    this.icon,
    this.fullWidth = false,
    this.isLoading = false,
    this.color,
    this.borderColor,
    this.textColor,
    this.showShadow = true,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        decoration: BoxDecoration(
          color: isActive ? const Color(0xFF3A86FF) : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
              color: isActive ? const Color(0xFF3A86FF) : Colors.white38),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(icon, color: Colors.white, size: 18),
              const SizedBox(width: 8),
            ],
            Text(label,
                style: GoogleFonts.inter(
                    color: Colors.white, fontWeight: FontWeight.w600)),
          ],
        ),
      ),
    );
  }
}

class AxonSkeleton extends StatelessWidget {
  final double? width;
  final double height;
  final double radius;

  const AxonSkeleton({
    super.key,
    this.width,
    required this.height,
    this.radius = 8,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      width: width,
      height: height,
      decoration: BoxDecoration(
        color: isDark ? Colors.white10 : Colors.black12,
        borderRadius: BorderRadius.circular(radius),
      ),
    );
  }
}

class AxonLogo extends StatelessWidget {
  final double size;

  const AxonLogo({super.key, this.size = 32});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF3A86FF), Color(0xFF8B5CF6)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        shape: BoxShape.circle,
      ),
      child: const Icon(Icons.psychology, color: Colors.white, size: 18),
    );
  }
}

class MetricCard extends StatelessWidget {
  final String? title;
  final String? value;
  final IconData? icon;
  final Color? color;
  final double? score;
  final String? label;
  final String? unit;
  final String? sublabel;
  final VoidCallback? onTap;

  const MetricCard({
    super.key,
    this.title,
    this.value,
    this.icon,
    this.color,
    this.score,
    this.label,
    this.unit,
    this.sublabel,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? const Color(0xFF1E1E2E) : Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: isDark ? Colors.white12 : Colors.black12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon ?? Icons.circle, color: color ?? const Color(0xFF3A86FF)),
          const SizedBox(height: 12),
          Text(value ?? '',
              style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white)),
          const SizedBox(height: 4),
          Text(title ?? '',
              style: TextStyle(
                  fontSize: 12, color: Colors.white.withValues(alpha: 0.6))),
        ],
      ),
    );
  }
}

class AnalogGauge extends StatelessWidget {
  final double value;
  final double maxValue;
  final String label;
  final Color? color;
  final double? score;
  final double? size;

  const AnalogGauge({
    super.key,
    this.value = 0,
    this.maxValue = 100,
    this.label = '',
    this.color,
    this.score,
    this.size,
  });

  @override
  Widget build(BuildContext context) {
    final gaugeColor = color ?? const Color(0xFF3A86FF);
    return SizedBox(
      height: 120,
      width: 120,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CircularProgressIndicator(
            value: value / maxValue,
            color: gaugeColor,
            backgroundColor: Colors.white12,
            strokeWidth: 8,
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(value.toInt().toString(),
                  style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white)),
              Text(label,
                  style: TextStyle(
                      fontSize: 10,
                      color: Colors.white.withValues(alpha: 0.6))),
            ],
          ),
        ],
      ),
    );
  }
}

class AxonBlockitSlider extends StatefulWidget {
  final double value;
  final ValueChanged<double> onChanged;
  final double min;
  final double max;
  final Color? activeColor;
  final int? steps;
  final VoidCallback? onTap;
  final String? title;
  final num? step;
  final Color? color;
  final String? label;
  final String Function(double)? valueFormatter;

  const AxonBlockitSlider({
    super.key,
    required this.value,
    required this.onChanged,
    this.min = 0,
    this.max = 100,
    this.activeColor,
    this.steps,
    this.onTap,
    this.title,
    this.step,
    this.color,
    this.label,
    this.valueFormatter,
  });

  @override
  State<AxonBlockitSlider> createState() => _AxonBlockitSliderState();
}

class _AxonBlockitSliderState extends State<AxonBlockitSlider> {
  @override
  Widget build(BuildContext context) {
    return Slider(
      value: widget.value,
      min: widget.min,
      max: widget.max,
      activeColor: widget.activeColor ?? const Color(0xFF3A86FF),
      onChanged: widget.onChanged,
    );
  }
}

class AxonGlassCard extends StatelessWidget {
  final Widget child;
  final double borderRadius;
  final double blur;
  final EdgeInsets? padding;
  final Color? glowColor;

  const AxonGlassCard({
    super.key,
    required this.child,
    this.borderRadius = 24,
    this.blur = 15,
    this.padding,
    this.glowColor,
  });

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      padding: padding ?? const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark ? Colors.white.withValues(alpha: 0.05) : Colors.white,
        borderRadius: BorderRadius.circular(borderRadius),
        border: Border.all(color: isDark ? Colors.white12 : Colors.black12),
        boxShadow: glowColor != null
            ? [
                BoxShadow(
                    color: glowColor!.withValues(alpha: 0.2), blurRadius: 20)
              ]
            : null,
      ),
      child: child,
    );
  }
}

class SectionHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? trailing;

  const SectionHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title,
            style: GoogleFonts.inter(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w600)),
        if (trailing != null) trailing!,
      ],
    );
  }
}

class AxonInput extends StatelessWidget {
  final Function(String)? onChanged;
  final String? hintText;
  final TextEditingController? controller;
  final bool obscureText;
  final TextInputType? keyboardType;
  final Widget? prefixIcon;
  final Widget? suffixIcon;

  const AxonInput({
    super.key,
    this.onChanged,
    this.hintText,
    this.controller,
    this.obscureText = false,
    this.keyboardType,
    this.prefixIcon,
    this.suffixIcon,
  });

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      obscureText: obscureText,
      keyboardType: keyboardType,
      onChanged: onChanged,
      style: GoogleFonts.inter(color: Colors.white),
      decoration: InputDecoration(
        hintText: hintText,
        hintStyle: GoogleFonts.inter(color: Colors.white38),
        prefixIcon: prefixIcon,
        suffixIcon: suffixIcon,
        filled: true,
        fillColor: Colors.white.withValues(alpha: 0.05),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white12),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white12),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Color(0xFF3A86FF)),
        ),
      ),
    );
  }
}
