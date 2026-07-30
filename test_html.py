
import json
frames = ['<svg><text>hello</text></svg>']
captions = ['hello']
base_delay = 20
frames_json = json.dumps(frames)
captions_json = json.dumps(captions)

html_code = '''
<!DOCTYPE html>
<html>
<body>
<script>
    const frames = __FRAMES__;
    const captions = __CAPTIONS__;
    const baseDelay = __BASE_DELAY__;
    console.log(frames);
</script>
</body>
</html>
'''.replace('__FRAMES__', frames_json).replace('__CAPTIONS__', captions_json).replace('__BASE_DELAY__', str(base_delay))

with open('test.html', 'w') as f:
    f.write(html_code)

