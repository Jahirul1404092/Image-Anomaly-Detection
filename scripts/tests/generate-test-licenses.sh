TEST_LIC_PATH=api/tests/licenses

# --- 有償版ライセンス ---
#   前提:
#     1. PyArmor のライセンス認証が行われていること。
#        まだの場合は、ライセンスファイルを取得し、以下のように認証を行う。
#
#        $ pyarmor register <Visee 用の認証ファイル>
#        $ pyarmor register
#          認証情報が出力されれば完了

# 制約なしのライセンス
LIC_NAME=valid \
    && pyarmor licenses ${LIC_NAME} \
    && cp licenses/${LIC_NAME}/license.lic ${TEST_LIC_PATH}/${LIC_NAME}.lic

# 常に期限切れとなるような過去日付を期限としたライセンス
LIC_NAME=expired \
    && pyarmor licenses ${LIC_NAME} \
        --expired 2000-01-01 \
    && cp licenses/${LIC_NAME}/license.lic ${TEST_LIC_PATH}/${LIC_NAME}.lic


# --- 無償版ライセンス ---
# 制約なしのライセンス
LIC_NAME=trial-valid \
    && pyarmor licenses ${LIC_NAME} \
    && cp licenses/${LIC_NAME}/license.lic ${TEST_LIC_PATH}/${LIC_NAME}.lic

# 常に期限切れとなるような過去日付を期限としたライセンス
LIC_NAME=trial-expired \
    && pyarmor licenses ${LIC_NAME} \
        --expired 2000-01-01 \
    && cp licenses/${LIC_NAME}/license.lic ${TEST_LIC_PATH}/${LIC_NAME}.lic
