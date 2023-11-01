# ライセンス認証方式について
※ 基本方針は[こちら](https://chowagiken.sharepoint.com/processdevelopment/SitePages/%E6%88%90%E6%9E%9C%E7%89%A9%E3%83%A9%E3%82%A4%E3%82%BB%E3%83%B3%E3%82%B9%E8%AA%8D%E8%A8%BC%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6.aspx?source=https%3A%2F%2Fchowagiken.sharepoint.com%2Fprocessdevelopment%2FSitePages%2FForms%2FByAuthor.aspx)を参照。


# ライセンス発行方法
1. PyArmor v7.4.4 をインストール
    - `dev` 環境にインストール済み
2. PyArmor 初期設定
    1. 1Password に登録されている「PyArmor (Visee)」のライセンスファイル `pyarmor-regfile-3106.zip` をDL
    2. 次のように実行し有償版ライセンスを有効化
        ```sh
        $ pyarmor register <Visee 用の認証ファイル>
        $ pyarmor register
          認証情報が出力されれば完了
        ```
3. 次のように期限を指定してライセンスを発行
    - `license/<ライセンス名>/license.lic` が生成される
    ```sh
    $ pyarmor licenses <ライセンス名> \
        --expired YYYY-MM-DD
    ```
4. `license/<ライセンス名>/license.lic` を顧客へ提供