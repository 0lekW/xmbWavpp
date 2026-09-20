// ============================================================================
//  XMB Wave — terminal ASCII port of the RetroArch "XMB Ribbon" shader.
//
//  This is a faithful reimplementation of the original THREE.js wallpaper
//  (js/index.js from the Wallpaper Engine item). The original renders a
//  128x128 plane whose vertices are displaced in a WebGL vertex shader, then
//  shaded in a fragment shader using screen-space derivatives.
//
//  Build:  g++ -O2 -std=c++17 -o waves waves.cpp
//  Run:    ./waves          (Ctrl-C to quit)
// ============================================================================

#include <cmath>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>

/* 
Terminal size / VT setup differs per platform. The primary target is a
Unix terminal (Linux/macOS/WSL); a Windows fallback is provided so the same
single file builds and runs everywhere. 
*/

#if defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <sys/ioctl.h>
  #include <unistd.h>
#endif

//  Small vector helper
struct Vec3 { float x, y, z; };

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
static inline float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static inline Vec3 normalize(const Vec3& v) {
    float m = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (m < 1e-8f) return {0.0f, 0.0f, 0.0f};
    return { v.x/m, v.y/m, v.z/m };
}

// GLSL built-ins used by the shader.
static inline float fract(float x) { return x - std::floor(x); }
static inline float mix(float a, float b, float t) { return a + t*(b - a); } // GLSL mix()

// Noise — ported verbatim from the original vertex shader.
static inline float iqhash(float n) { return fract(std::sin(n) * 43758.5453f); }
static float noise(const Vec3& x) {
    Vec3 p { std::floor(x.x), std::floor(x.y), std::floor(x.z) };
    Vec3 f { fract(x.x), fract(x.y), fract(x.z) };
    f = { f.x*f.x*(3.0f - 2.0f*f.x),
          f.y*f.y*(3.0f - 2.0f*f.y),
          f.z*f.z*(3.0f - 2.0f*f.z) };
    float n = p.x + p.y*57.0f + 113.0f*p.z;
    return mix(mix(mix(iqhash(n),        iqhash(n + 1.0f),   f.x),
                   mix(iqhash(n + 57.0f),  iqhash(n + 58.0f),  f.x), f.y),
               mix(mix(iqhash(n + 113.0f), iqhash(n + 114.0f), f.x),
                   mix(iqhash(n + 170.0f), iqhash(n + 171.0f), f.x), f.y), f.z);
}

static inline float xmb_noise2(const Vec3& x, float time) {
    return std::cos(x.z * 4.0f) * std::cos(x.z + time / 10.0f + x.x);
}

// ---------------------------------------------------------------------------
//  Base plane extents.
//
//  The original builds the ribbon from the *projected* plane:
//    vec4 pos = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
//    vec3 v   = vec3(pos.x, 0.0, pos.y);
//  with PlaneGeometry(1,1,128,128)  -> position in [-0.5, 0.5]^2
//       ribbon.scale = (aspect * 1.55, 0.75, 1)
//       PerspectiveCamera(75, aspect, ...), camera.z = 2
//
//  Working the perspective projection through by hand, the aspect ratio
//  cancels out (the x-scale multiplies by aspect, the projection divides by
//  it), leaving a fixed clip-space plane:
//     pos.x = f * 1.55 * px,   pos.y = f * 0.75 * py
//  where f = 1 / tan(fov/2), fov = 75deg, and px,py in [-0.5, 0.5].
//
//  So the ribbon's base coordinates are independent of terminal aspect:
//     v.x in ~[-1.01, 1.01],  v.z in ~[-0.489, 0.489].
// ---------------------------------------------------------------------------
static const float FOV_DEG   = 75.0f;
static const float FOCAL     = 1.0f / std::tan((FOV_DEG * 0.5f) * 3.14159265358979323846f / 180.0f);
static const float PLANE_X   = FOCAL * 1.55f;   // half-extent multiplier for v.x
static const float PLANE_Z   = FOCAL * 0.75f;   // half-extent multiplier for v.z


//  Vertex displacement — ported verbatim from the vertex shader main()
static Vec3 ribbon_vertex(float px, float py, float time) {
    // vec3 v = vec3(pos.x, 0.0, pos.y);
    Vec3 v { PLANE_X * px, 0.0f, PLANE_Z * py };
    Vec3 v2 = v;
    Vec3 v3 = v;

    // v.y = xmb_noise2(v2) / 8.0;
    v.y = xmb_noise2(v2, time) / 8.0f;

    // v3.x -= time / 5.0;  v3.x /= 4.0;
    v3.x -= time / 5.0f;
    v3.x /= 4.0f;
    // v3.z -= time / 10.0;  v3.y -= time / 100.0;
    v3.z -= time / 10.0f;
    v3.y -= time / 100.0f;

    // noise(v3 * 7.0) — evaluated once, used twice below.
    float n = noise(Vec3{ v3.x * 7.0f, v3.y * 7.0f, v3.z * 7.0f });

    // v.z -= noise(v3 * 7.0) / 15.0;
    v.z -= n / 15.0f;
    // v.y -= noise(v3 * 7.0) / 15.0 + cos(v.x * 2.0 - time / 2.0) / 5.0 - 0.3;
    v.y -= n / 15.0f + std::cos(v.x * 2.0f - time / 2.0f) / 5.0f - 0.3f;

    return v; // == vEC (the varying passed to the fragment shader)
}

//  Fragment shading — ported from the fragment shader.
static inline float ribbon_alpha(const Vec3& ddx, const Vec3& ddy) {
    const Vec3 up { 0.0f, 0.0f, 1.0f };
    Vec3 normal = normalize(cross(ddx, ddy));
    float c = 1.0f - dot(normal, up);
    c = (1.0f - std::cos(c * c)) / 3.0f;
    return c * 1.5f; // gl_FragColor.a
}

// ---------------------------------------------------------------------------
//  Presentation constants. The math above is exact; these only control how
//  the fixed clip-space ribbon is mapped into terminal cells (the browser
//  version fills the viewport, here we choose a framing).
// ---------------------------------------------------------------------------
static const float NDC_HALF_WIDTH = 1.05f;  // v.x range mapped across the width
static const float V_CENTER       = 0.27f;  // approx. mean of v.y (ribbon center)
static const float V_GAIN         = 1.9f;   // vertical amplification (terminal cells are tall)

// ASCII intensity ramp (dark -> bright), matching the alpha of the ribbon.
static const char* SHADES  = " .:-=+*#%@";
static const int   NSHADES = 10;

// ---------------------------------------------------------------------------
//  Terminal handling.
// ---------------------------------------------------------------------------
static void get_term_size(int& w, int& h) {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        w = csbi.srWindow.Right  - csbi.srWindow.Left + 1;
        h = csbi.srWindow.Bottom - csbi.srWindow.Top  + 1;
    } else {
        w = 80; h = 24;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        w = ws.ws_col;
        h = ws.ws_row;
    } else {
        w = 80; h = 24;
    }
#endif
}

// On Windows, opt the console into ANSI/VT escape-sequence processing so the
// same escape codes used on Unix work here too. No-op elsewhere.
static void enable_vt_mode() {
#if defined(_WIN32)
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(hOut, &mode))
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    SetConsoleOutputCP(CP_UTF8);
#endif
}

static volatile std::sig_atomic_t g_running = 1;
static void on_sigint(int) { g_running = 0; }

int main() {
    std::signal(SIGINT,  on_sigint);
#ifdef SIGTERM
    std::signal(SIGTERM, on_sigint);
#endif
    enable_vt_mode();

    int W = 0, H = 0;
    get_term_size(W, H);

    // ORIGINAL: ribbon.material.uniforms.time.value starts at 1.0.
    float time = 1.0f;
    const float TIME_STEP = 0.01f;                      // matches uniforms.time += 0.01
    const auto  FRAME = std::chrono::milliseconds(16);  // ~60 fps, like requestAnimationFrame

    std::vector<float> bright;
    std::string frame;

    std::printf("\x1b[?25l");   // hide cursor
    std::printf("\x1b[2J");     // clear screen

    while (g_running) {
        // Handle terminal resizes on the fly.
        int nw, nh;
        get_term_size(nw, nh);
        if (nw != W || nh != H) {
            W = nw; H = nh;
            std::printf("\x1b[2J");
        }
        if (W < 2 || H < 2) { std::this_thread::sleep_for(FRAME); continue; }

        bright.assign((size_t)W * H, 0.0f);

        // Sample density: enough columns to cover the width, and enough rows
        // per column so the swept band has no vertical gaps.
        const int MESH_X = std::max(W * 2, 64);
        const int MESH_Z = std::max(H * 8, 256);
        const float EPS  = 1.0f / 1024.0f; // finite-difference step in plane params

        for (int ix = 0; ix < MESH_X; ++ix) {
            float px = (float)ix / (float)(MESH_X - 1) - 0.5f; // [-0.5, 0.5]
            for (int iz = 0; iz < MESH_Z; ++iz) {
                float py = (float)iz / (float)(MESH_Z - 1) - 0.5f; // [-0.5, 0.5]

                Vec3 p  = ribbon_vertex(px,       py,       time);
                Vec3 dx = ribbon_vertex(px + EPS, py,       time);
                Vec3 dy = ribbon_vertex(px,       py + EPS, time);

                float alpha = ribbon_alpha(
                    Vec3{ dx.x - p.x, dx.y - p.y, dx.z - p.z },
                    Vec3{ dy.x - p.x, dy.y - p.y, dy.z - p.z });
                if (alpha <= 0.0f) continue;

                // NDC (v.xy) -> screen cell.
                float sxf = ( p.x / NDC_HALF_WIDTH * 0.5f + 0.5f) * (W - 1);
                float syf = ( 0.5f - (p.y - V_CENTER) * V_GAIN * 0.5f) * (H - 1);
                int sx = (int)(sxf + 0.5f);
                int sy = (int)(syf + 0.5f);
                if (sx < 0 || sx >= W || sy < 0 || sy >= H) continue;

                // Alpha-blended white over background -> keep the strongest.
                float& b = bright[(size_t)sy * W + sx];
                if (alpha > b) b = alpha;
            }
        }

        // Compose the frame.
        frame.clear();
        frame.reserve((size_t)W * H * 20 + H * 8);
        frame += "\x1b[H"; // cursor home

        int lastFg = -1;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float b = bright[(size_t)y * W + x];
                float bc = std::min(std::max(b, 0.0f), 1.0f);

                if (bc <= 0.02f) {
                    frame += ' '; // transparent — the terminal's own background
                    continue;
                }

                // Ribbon cell: white with brightness from alpha.
                int gray = 232 + (int)(bc * 23.0f + 0.5f); // 232..255 grayscale ramp
                int shadeIdx = (int)(bc * (NSHADES - 1) + 0.5f);
                if (shadeIdx < 0) shadeIdx = 0;
                if (shadeIdx >= NSHADES) shadeIdx = NSHADES - 1;

                if (gray != lastFg) {
                    char buf[24];
                    std::snprintf(buf, sizeof(buf), "\x1b[38;5;%dm", gray);
                    frame += buf;
                    lastFg = gray;
                }
                frame += SHADES[shadeIdx];
            }
            frame += "\x1b[0m";
            lastFg = -1;
            if (y != H - 1) frame += '\n';
        }

        std::fwrite(frame.data(), 1, frame.size(), stdout);
        std::fflush(stdout);

        time += TIME_STEP;
        std::this_thread::sleep_for(FRAME);
    }

    std::printf("\x1b[0m\x1b[?25h\n"); // reset colors, show cursor
    std::fflush(stdout);
    return 0;
}
