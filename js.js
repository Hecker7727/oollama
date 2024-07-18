function loadScript(url, callback) {
    var head = document.head;
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = url;
    script.async = true;
    script.onreadystatechange = callback;
    script.onload = callback;
    head.appendChild(script);
  }
  
  var options = {
    "loader": "unity2022",
    "loaderOptions": {
      "showProgress": true,
      "unityLoaderUrl": "https://files.crazygames.com/hazmob-fps-online-shooter/16/34/Build/bb0d9ecdb05db3e84da20bd14a4f84dc.loader.js",
      "unityConfigOptions": {
        "codeUrl": "https://files.crazygames.com/hazmob-fps-online-shooter/16/34/Build/39bf930fdb625f2d286f81b557ea590f.wasm.br",
        "dataUrl": "https://files.crazygames.com/hazmob-fps-online-shooter/16/34/Build/8a767108ea83aaeca2a9b51723b6c2e3.data.br",
        "frameworkUrl": "https://files.crazygames.com/hazmob-fps-online-shooter/16/34/Build/9e4df48d8b185b696e1b87367dad3da2.framework.js.br",
        "streamingAssetsUrl": "https://files.crazygames.com/hazmob-fps-online-shooter/16/34/StreamingAssets"
      }
    },
    "thumbnail": "hazmob-fps-online-shooter_16x9/20240607181337/hazmob-fps-online-shooter_16x9-cover",
    "gameName": "Hazmob FPS: Online Shooter",
    "gameHttps": true,
    "gameSlug": "hazmob-fps-online-shooter",
    "gameId": "22513",
    "gameStatus": "published",
    "showAdOnExternal": "ALWAYS",
    "showAdOnInternal": "DEFAULT",
    "disableEmbedding": false,
    "dollarRate": 0.9177300864042877,
    "upvotes": 138708,
    "downvotes": 21465,
    "releaseDate": "2024-02-26",
    "locale": "en_US",
    "controls": {
      "text": "<h2>Controls</h2>\n<ul>\n    <li>WASD = move</li>\n    <li>Space = jump</li>\n    <li>Left mouse = shoot</li>\n    <li>Right mouse / V = aim</li>\n    <li>P = leaderboard, pause, and settings</li>\n    <li>G = pick up the gun</li>\n    <li>C = crouch</li>\n    <li>Shift = run</li>\n    <li>1, 2, 3 = switch weapons</li>\n    <li>4, 5, 6 = skills</li>\n    <li>E = alternative skills</li>\n    <li>Q = alternative skills </li>\n</ul>"
    },
    "videoThumbnail": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-205x115.mp4",
    "video": {
      "original": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter.mp4",
      "sizes": [
        {"width": 205, "height": 115, "location": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-205x115.mp4"},
        {"width": 249, "height": 140, "location": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-249x140.mp4"},
        {"width": 364, "height": 208, "location": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-364x208.mp4"},
        {"width": 494, "height": 276, "location": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-494x276.mp4"},
        {"width": 611, "height": 343, "location": "https://videos.crazygames.com/hazmob-fps-online-shooter/3/hazmob-fps-online-shooter-611x343.mp4"}
      ]
    },
    "sandbox": true,
    "totalSizeBytes": 141808778,
    "minTimeBetweenMidrollAdsInMs": 180000,
    "minTimeBetweenRewardedAdsInMs": 5000,
    "noMidrollAdInFirstMinutes": 3,
    "noRewardedAdInFirstMinutes": 0,
    "categoryEnSlug": "shooting",
    "tagsEnSlugs": ["arena", "war", "gun", "party", "multiplayer", "first-person-shooter"],
    "isKids": false,
    "hasMidroll": true,
    "hasRewarded": true,
    "showExternalProviderWarning": false,
    "aps": "in-game-auto-login",
    "fullscreen": "ADDS_VALUE",
    "orientation": null,
    "apsStorageType": "disabled",
    "userProgressionSaveMessage": "cloud",
    "category": "FPS",
    "categoryLink": "https://www.crazygames.com/t/first-person-shooter",
    "gameLink": "https://www.crazygames.com/game/hazmob-fps-online-shooter",
    "domainHasNoCMP": false,
    "source": "games"
  };
  
  var SDK_READY = false;
  var GF_READY = false;
  var GF_LOAD_CALLED = false;
  
  // Build GF path
  var gfBuildPath = 'https://builds.crazygames.com/gameframe'
  var gfDefaultVersion = '1'
  var useLocalGF = false
  var params = new URLSearchParams(window.location.search);
  var version = params.get('v');
  // Validate version param to prevent XSS
  if(version) {
    var versionRegex = /^\d+(\.\d+)?(\.\d+)?$/;
    if(!versionRegex.test(version)) {
      console.error('Invalid Gameframe version provided, falling back to default version');
      version = null;
    }
  }
  var gameframeJs = gfBuildPath + '/v' + (version || gfDefaultVersion) + '/bundle.js';
  if (useLocalGF) {
    gameframeJs = 'http://localhost:3002/static/js/bundle.js';
  }
  
  loadScript(gameframeJs, function() {
    Crazygames.load(options).then(function() {
      var loaderWrapper = document.getElementById('ziggyLoader');
      var loaderStyles = document.getElementById('ziggyLoaderStyles');
      if (loaderWrapper && loaderStyles) {
        loaderWrapper.remove();
        loaderStyles.remove();
      }
    });   
  });
  