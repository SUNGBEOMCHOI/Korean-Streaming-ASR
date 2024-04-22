from src.utils.transcriber import DenoiseTranscriber
from src.utils.argparse_config import setup_arg_parser

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    transcriber = DenoiseTranscriber(args)

    if args.inference:
        transcriber.transcribe(manifest_path=args.manifest_path, inference_result_path=args.inference_result_path)
    else:
        if args.mode == 'microphone':
            transcriber.transcribe()
        elif args.mode == 'file':
            transcriber.transcribe(audio_path=args.audio_path)
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
