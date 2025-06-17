#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

// camera (eye) position
const float EYE_X =  0.0f;
const float EYE_Y = 0.0f;
const float EYE_Z = -20.0f;

// point to look at
const float TARGET_X = 0.0f;
const float TARGET_Y = 0.0f;
const float TARGET_Z = 0.0f;

// “up” direction
const float UP_X = 0.0f;
const float UP_Y = 1.0f;
const float UP_Z = 0.0f;

// focal length
const float FOCAL = 90.0f;

const float ZOOM_OUT_SCALE = 1.5f;

const char* SHADES  = " .:-=+*#%@";
const int   NSHADES = std::strlen(SHADES);

struct Vec3 { float x, y, z; };
static Vec3 normalize(const Vec3& v) {
    float m = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    return { v.x/m, v.y/m, v.z/m };
}
static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    };
}
static float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

// ——— Improved Perlin 3D noise ———
static const int PERM[512] = {
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,
    197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,
    56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,
    27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,
    92,41,55,46,245,40,244,102,143,54, 65,25,63,161, 1,216,
    80,73,209,76,132,187,208, 89,18,169,200,196,135,130,116,188,
    159,86,164,100,109,198,173,186,  3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,
    58,17,182,189,28,42,223,183,170,213,119,248,152,  2,44,154,
    163, 70,221,153,101,155,167,  43,172,9,129,22,39,253, 19,98,
    108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,
    242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,
    239,107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,
    50,45,127,  4,150,254,138,236,205, 93,222,114, 67,29,24,72,
    243,141,128,195,78,66,215,61,156,180,
    // repeat again:
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,
    197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,
    56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,
    27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,
    92,41,55,46,245,40,244,102,143,54, 65,25,63,161, 1,216,
    80,73,209,76,132,187,208, 89,18,169,200,196,135,130,116,188,
    159,86,164,100,109,198,173,186,  3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,
    58,17,182,189,28,42,223,183,170,213,119,248,152,  2,44,154,
    163, 70,221,153,101,155,167,  43,172,9,129,22,39,253, 19,98,
    108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,34,
    242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,
    239,107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,
    50,45,127,  4,150,254,138,236,205, 93,222,114, 67,29,24,72,
    243,141,128,195,78,66,215,61,156,180
};

static float fadef(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
static float lerpf(float a, float b, float t) { return a + t*(b - a); }
static float grad(int hash, float x, float y, float z) {
    int h = hash & 15;          
    float u = h<8 ? x : y;      
    float v = h<4 ? y : (h==12||h==14 ? x : z);
    return ((h&1)? -u : u) + ((h&2)? -v : v);
}

float perlin3d(float x, float y, float z) {
    int X = int(std::floor(x)) & 255;
    int Y = int(std::floor(y)) & 255;
    int Z = int(std::floor(z)) & 255;
    x -= std::floor(x); y -= std::floor(y); z -= std::floor(z);
    float u = fadef(x), v = fadef(y), w = fadef(z);

    int A  = PERM[X] + Y,   AA = PERM[A] + Z,   AB = PERM[A + 1] + Z;
    int B  = PERM[X+1] + Y, BA = PERM[B] + Z,   BB = PERM[B + 1] + Z;

    float res = lerpf(
      lerpf(
        lerpf( grad(PERM[AA],   x,   y,   z),
               grad(PERM[BA], x-1.0f,y,   z), u),
        lerpf( grad(PERM[AB],   x, y-1.0f,   z),
               grad(PERM[BB], x-1.0f,y-1.0f,  z), u),
         v),
      lerpf(
        lerpf( grad(PERM[AA+1],   x,   y,   z-1.0f),
               grad(PERM[BA+1], x-1.0f,y,   z-1.0f), u),
        lerpf( grad(PERM[AB+1],   x, y-1.0f,   z-1.0f),
               grad(PERM[BB+1], x-1.0f,y-1.0f, z-1.0f), u),
         v),
      w);
    return res; // in [-1..1]
}

// Simple 3D hash noise
inline float fract(float x) { return x - std::floor(x); }
inline float lerp(float a, float b, float t) { return a + t*(b - a); }
inline float iqhash(float n) { return fract(std::sin(n) * 43758.5453f); }

float noise(const Vec3& x) {
    Vec3 p{std::floor(x.x), std::floor(x.y), std::floor(x.z)};
    Vec3 f{fract(x.x), fract(x.y), fract(x.z)};
    f = { f.x*f.x*(3.0f - 2.0f*f.x), f.y*f.y*(3.0f - 2.0f*f.y), f.z*f.z*(3.0f - 2.0f*f.z) };
    float n = p.x + p.y*57.0f + p.z*113.0f;
    float m0 = lerp(iqhash(n),       iqhash(n+1.0f),   f.x);
    float m1 = lerp(iqhash(n+57.0f), iqhash(n+58.0f),  f.x);
    float m2 = lerp(iqhash(n+113.0f),iqhash(n+114.0f), f.x);
    float m3 = lerp(iqhash(n+170.0f),iqhash(n+171.0f), f.x);
    float a = lerp(m0, m1, f.y);
    float b = lerp(m2, m3, f.y);
    return lerp(a, b, f.z);
}

// trig-based noise2
inline float noise2(float x, float z, float t) {
    return std::cos(z * 4.0f) * std::cos(z + t/10.0f + x);
}

// directional light direction
const Vec3 LIGHT_DIR = normalize(Vec3{0.0f, 1.0f, 1.0f});

// evaluate the surface point at (x,z)
Vec3 eval_surface(float x, float z, float t) {
    float y1 = std::sin(x * 0.3f + t);
    float y2 = std::cos(z * 0.3f + t * 1.5f);
    Vec3 p = { x, y1 + y2, z };
    Vec3 d2 = { (p.x - t/5.0f) / 4.0f, -t/100.0f, (p.z - t/10.0f) / 4.0f };
    //float n1 = noise(Vec3{d2.x*7.0f, d2.y*7.0f, d2.z*7.0f}) / 15.0f;
    float n1 = perlin3d(d2.x*0.7, d2.y*0.7, d2.z*0.7) * 0.1;
    float n2 = noise2(d2.x, d2.z, t) * 0.2f;
    p.y -= (n1 + n2);
    p.z -= (n1 + n2) * 0.5f;
    return p;
}

int main() {
    float t = 0.0f;
    const float dt = 0.1f;

    // detect terminal size
    struct winsize ws;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    int WIDTH  = ws.ws_col;
    int HEIGHT = ws.ws_row > 2 ? ws.ws_row - 2 : ws.ws_row;

    int MESH_X = WIDTH * 2;
    int MESH_Z = HEIGHT * 2;

    // build camera basis
    Vec3 eye    = {EYE_X, EYE_Y, EYE_Z};
    Vec3 target = {TARGET_X, TARGET_Y, TARGET_Z};
    Vec3 updir  = {UP_X, UP_Y, UP_Z};
    Vec3 fwd    = normalize(Vec3{ target.x - eye.x, target.y - eye.y, target.z - eye.z });
    Vec3 right  = normalize(cross(fwd, updir));
    Vec3 up     = cross(right, fwd);

    // compute plane size
    Vec3 te = { target.x - eye.x, target.y - eye.y, target.z - eye.z };
    float cz0 = dot(te, fwd);
    float fullPlaneD = cz0 * HEIGHT / FOCAL;
    float halfD = fullPlaneD * 0.5f;
    float cz_near = cz0 - halfD;
    float planeW = cz_near * WIDTH  / FOCAL * ZOOM_OUT_SCALE;
    float planeD = fullPlaneD       * ZOOM_OUT_SCALE;

    float NEAR_CLIP = cz0 - planeD * 0.5f;
    float FAR_CLIP  = cz0 + planeD * 0.5f;

    float dx = planeW / (MESH_X - 1);
    float dz = planeD / (MESH_Z - 1);

    // buffers
    std::vector<char>  screen(WIDTH * HEIGHT);
    std::vector<float> zbuf  (WIDTH * HEIGHT);
    std::vector<float> bright(WIDTH * HEIGHT);

    std::cout << "\x1b[2J";
    while (true) {
        std::fill(screen.begin(), screen.end(), ' ');
        std::fill(zbuf.begin(),   zbuf.end(),   1e9f);

        for (int ix = 0; ix < MESH_X; ++ix) {
            float x = (ix/(float)(MESH_X-1) - 0.5f) * planeW;
            for (int iz = 0; iz < MESH_Z; ++iz) {
                float z = (iz/(float)(MESH_Z-1) - 0.5f) * planeD;
                // sample surface
                Vec3 p  = eval_surface(x,     z,     t);
                Vec3 pR = eval_surface(x+dx,  z,     t);
                Vec3 pD = eval_surface(x,     z+dz,  t);
                // normal & lighting
                Vec3 N = normalize(cross(
                      Vec3{pD.x-p.x, pD.y-p.y, pD.z-p.z},
                      Vec3{pR.x-p.x, pR.y-p.y, pR.z-p.z}
                      ));
                float rawL = dot(N, LIGHT_DIR);
                const float AMBIENT = 0.0f;
                float L = AMBIENT + (1 - AMBIENT) * rawL;
                L = std::min(std::max(L, 0.0f), 1.0f);


                // camera space
                Vec3 v_cam = { p.x - eye.x, p.y - eye.y, p.z - eye.z };
                float cz    = dot(v_cam, fwd);
                if (cz <= 0.1f) continue;
                float proj = FOCAL / cz;
                int sx = int(dot(v_cam, right) * proj + WIDTH/2);
                int sy = int(HEIGHT/2 - dot(v_cam, up) * proj);
                if (sx < 0 || sx >= WIDTH || sy < 0 || sy >= HEIGHT) continue;

                int idx = sy * WIDTH + sx;
                if (cz < zbuf[idx]) {
                    zbuf[idx] = cz;
                    bright[idx] = L;
                    int shadeIdx = int((1.0f - L) * (NSHADES - 1));
                    screen[idx] = SHADES[shadeIdx];
                }
            }
        }


        // draw frame with 256-color grayscale
        std::cout << "\x1b[H";  // cursor home
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                int idx = y*WIDTH + x;

                // map bright[idx] ∈ [0..1] → gray level 0..23
                float bclamped = std::min(std::max(bright[idx], 0.0f), 1.0f);
                int   grayLevel = int(bclamped * 23.0f + 0.5f);
                int   ansiCode  = 232 + grayLevel;      // 232..255

                // set FG to that gray, then print your char (or block)
                std::cout
                  << "\x1b[38;5;" << ansiCode << "m"
                  << screen[idx];
            }
            // reset color and newline
            std::cout << "\x1b[0m\n";
        }
        std::cout.flush();

        t += dt;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
