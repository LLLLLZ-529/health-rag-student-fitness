#!/usr/bin/env python3
"""HealthAgent CLI 入口
用法：
  python run.py --query "帮我看看这个学生该怎么练" --bmi 22 --vc 2400 --str 8 --endurance 300 --speed 9.5 --jump 150 --flex 3
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import get_app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--bmi", type=float, default=22)
    ap.add_argument("--vc", type=int, default=2800)
    ap.add_argument("--str", type=float, default=15)
    ap.add_argument("--endurance", type=int, default=270, help="耐力跑秒数")
    ap.add_argument("--speed", type=float, default=9.0)
    ap.add_argument("--jump", type=int, default=170)
    ap.add_argument("--flex", type=float, default=10)
    ap.add_argument("--history", type=str, default=None, help="JSON 字符串：历年 HI")
    args = ap.parse_args()

    student_data = {
        "bmi": args.bmi, "vc": args.vc, "str": args.str,
        "endurance": args.endurance, "speed": args.speed,
        "jump": args.jump, "flex": args.flex,
    }
    history = json.loads(args.history) if args.history else []

    app = get_app()
    result = app.invoke({
        "user_query": args.query,
        "student_data": student_data,
        "user_history": history,
    })

    print("\n" + "="*60)
    print(result.get("final_response", "无输出"))
    print("="*60)
    print("\n执行轨迹:")
    for step in result.get("trace", []):
        print(f"  [{step.get('node')}] {step}")

if __name__ == "__main__":
    main()
