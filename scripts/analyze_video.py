"""
Script para analizar videos y calcular métricas de detección de logos.

Ejemplo de uso:
    python scripts/analyze_video.py --video path/to/video.mp4
    python scripts/analyze_video.py --video video.mp4 --model runs/detect/runs/logo_detection/full_training_run/weights/best.pt
    python scripts/analyze_video.py --video video.mp4 --extract-frames --no-video

Métricas calculadas:
    - Tiempo total de aparición por clase/logo
    - Porcentaje de frames con detecciones
    - Frames extraídos con detecciones anotadas
    - Reporte JSON con métricas detalladas
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.video_analytics import VideoAnalyzer, analyze_video
from src.config import OUTPUT_DIR, DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(
        description='Analizar video para detección de logos con métricas detalladas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --video mi_video.mp4
  %(prog)s --video mi_video.mp4 --model best.pt --conf 0.5
  %(prog)s --video mi_video.mp4 --extract-every 5 --max-frames 100
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Ruta al archivo de video a analizar'
    )
    
    # Model settings
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Ruta al modelo YOLO (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Umbral de confianza para detecciones (default: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='Umbral IOU para NMS (default: 0.45)'
    )
    
    # Output settings
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=f'Directorio de salida (default: {OUTPUT_DIR})'
    )
    
    # Frame extraction settings
    parser.add_argument(
        '--extract-frames',
        action='store_true',
        default=True,
        help='Extraer frames con detecciones (default: True)'
    )
    
    parser.add_argument(
        '--no-extract-frames',
        action='store_true',
        help='No extraer frames con detecciones'
    )
    
    parser.add_argument(
        '--extract-every',
        type=int,
        default=1,
        help='Extraer cada N frames con detecciones (default: 1 = todos)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Máximo número de frames a extraer (default: sin límite)'
    )
    
    # Video output settings
    parser.add_argument(
        '--save-video',
        action='store_true',
        default=True,
        help='Guardar video anotado (default: True)'
    )
    
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='No guardar video anotado'
    )
    
    # Report settings
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='No guardar reporte JSON'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video no encontrado: {video_path}")
        sys.exit(1)
    
    # Configure output directory
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    
    # Configure extraction and video saving
    extract_frames = args.extract_frames and not args.no_extract_frames
    save_video = args.save_video and not args.no_video
    
    print("=" * 60)
    print("ANÁLISIS DE VIDEO - DETECCIÓN DE LOGOS")
    print("=" * 60)
    print(f"\n📹 Video: {video_path}")
    print(f"🤖 Modelo: {args.model}")
    print(f"🎯 Confianza: {args.conf}")
    print(f"📁 Output: {output_dir}")
    print(f"📷 Extraer frames: {'Sí' if extract_frames else 'No'}")
    print(f"🎬 Guardar video: {'Sí' if save_video else 'No'}")
    print()
    
    # Create analyzer
    analyzer = VideoAnalyzer(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run analysis
    try:
        result = analyzer.analyze_video(
            video_path=video_path,
            output_dir=output_dir,
            extract_frames=extract_frames,
            save_annotated_video=save_video,
            extract_every_n_frames=args.extract_every,
            max_extracted_frames=args.max_frames
        )
        
        # Save report
        if not args.no_report:
            report_path = analyzer.save_report(result)
            print(f"\n✅ Análisis completado exitosamente!")
        else:
            print(f"\n✅ Análisis completado (sin reporte JSON)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
