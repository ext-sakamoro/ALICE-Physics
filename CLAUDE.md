# ALICE-Physics — Claude Code 設定

## プロジェクト概要

Deterministic 128-bit Fixed-Point Physics Engine

| 項目 | 値 |
|------|-----|
| クレート名 | `alice-physics` |
| バージョン | 0.6.0 |
| ライセンス | AGPL-3.0 |
| リポジトリ | `ext-sakamoro/ALICE-Physics` |
| Eco-Systemブリッジ | bridge_physics.rs + bridge_physics_2d.rs + bridge_physics_scene_io.rs + bridge_physics_softbody.rs |

## コーディングルール

メインCLAUDE.md「Git Commit設定」参照。日本語コミット・コメント、署名禁止、作成者 `Moroya Sakamoto`。

## ALICE 品質基準

ALICE-KARIKARI.md「100/100品質基準」参照。clippy基準: `pedantic+nursery`

| 指標 | 値 |
|------|-----|
| clippy (pedantic+nursery) | 0 warnings |
| テスト数 | 659 |
| fmt | clean |

## Eco-System パイプライン

本クレートはALICE-Eco-Systemの以下のパスで使用:
- Path B (Game→Physics)
- Path C (MoCap→Physics)
- Path G (AI→Physics)

## 情報更新ルール

- バージョンアップ時: このCLAUDE.mdのバージョンを更新
- APIの破壊的変更時: ALICE-Eco-Systemブリッジへの影響をメモ
- テスト数/品質の変化時: 品質基準セクションを更新
- 新feature追加時: プロジェクト概要テーブルを更新
