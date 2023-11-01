# Usages:
#   前提:
#     1. PyArmor のライセンス認証が行われていること。
#        まだの場合は、ライセンスファイルを取得し、以下のように認証を行う。
#
#        $ pyarmor register <Visee 用の認証ファイル>
#        $ pyarmor register
#          認証情報が出力されれば完了
#
#     2. 環境変数 OBFUSCATED_VERIFIER_DUMMY_PATH に難読化された verifier_dummy.py
#        の出力先パスが設定されていること。
OBFUSCATED_VERIFIER_DUMMY_DIR=obfuscated \
&& pyarmor obfuscate \
  --exact \
  --with-license outer \
  --output ${OBFUSCATED_VERIFIER_DUMMY_DIR} \
  api/util/verifier_dummy.py