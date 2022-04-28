STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
with open("../css/page.css", "r") as t:
  CSS = t.read()
with open("../js/page.js", "r") as t:
  JS = t.read()
WEB_HTML = """<html>
    <link rel="stylesheet" href={} />
    <style>
    {}
    </style>
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns" style="height: 100%">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 100%">
                <div id="parent" class="hero-body" style="overflow: auto; height: calc(100% - 76px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>対話のページ</strong>
                            <br>
                            システムと対話をするためのページです．<br>
                            対話をやり直す場合はページをリロードしてください．<br>
                        </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth" style="height: 76px">
                  <form id = "interact">
                      <div class="field is-grouped">
                        <p class="control is-expanded">
                          <input class="input" type="text" id="userIn" placeholder="メッセージを入力してください" required minlength="5" maxlength="120">
                          <input type="text" name="dummy" style="display:none;">
                          <span class="validity"></span>
                        </p>
                        <p class="control">
                          <button id="respond" type="button" onclick="next();" class="button has-text-white-ter has-background-grey-dark">
                            入力
                          </button>
                        </p>
                      </div>
                  </form>
                </div>
              </section>
            </div>
        </div>
        <script>
        {}
        </script>
    </style>
    </body>
</html>
"""  # noqa: E501